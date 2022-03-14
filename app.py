import h3.unstable.vect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pyArango.connection import Connection, Database
import os
import rasterio
from arango_queries import get_hotspot_dict, get_witnesses_for_hotspot, get_witnesses_of_hotspot
from gis_utils import get_local_elevation_map
from feature_extraction import process_witness_paths, get_bearing
import numpy as np
from haversine import haversine, inverse_haversine, Unit


load_dotenv()


TRAINED_SVM_PATH = "static/trained_models/svm/2022-02-06T16_23_54.mdl"
TRAINED_GP_PATH = "static/trained_models/gaussian_process/2022-02-04T16_28_14.mdl"
TRAINED_ISO_PATH = "static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl"


@st.experimental_singleton
def load_model(path: str):
    print("Loading trained model...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("done.")
    return model


svm = load_model(TRAINED_SVM_PATH)
gp = load_model(TRAINED_GP_PATH)
iso_forest = load_model(TRAINED_ISO_PATH)

# put db credentials in .env file
load_dotenv()

try:
    c = Connection(
        arangoURL=os.getenv('ARANGO_URL'),
        username=os.getenv('ARANGO_USERNAME'),
        password=os.getenv('ARANGO_PASSWORD')
    )
except ConnectionError:
    raise Exception('Unable to connect to the ArangoDB instance. Please check that it is running and that you have supplied the correct URL/credentials in the .env file.')
db: Database = c['helium-graphs']


# EVALUATION
def generate_features(db: Database, hotspot_address: str):
    hotspot_dict = get_hotspot_dict(db, hotspot_address)
    # witness_paths = get_witnesses_for_hotspot(db, hotspot_address, limit=1000)
    witness_paths = get_witnesses_of_hotspot(db, hotspot_address, limit=1000)


    with rasterio.open(os.getenv("VRT_PATH")) as dataset:
        elevation_map, window = get_local_elevation_map(dataset, hotspot_dict["latitude"], hotspot_dict["longitude"], range_km=250)
        elevation_map[elevation_map == dataset.nodata] = 0
        path_features, path_details, witness_coords, profiles = process_witness_paths(witness_paths, dataset, elevation_map, window, return_coords=True)

    return pd.DataFrame(path_features), pd.DataFrame(path_details), witness_coords, profiles


def find_outliers(features_df: pd.DataFrame, details_df: pd.DataFrame, iso_forest):
    predictions = iso_forest.predict(features_df)
    scores = iso_forest.decision_function(features_df)

    # isolation score < 0 indicates outliers
    output_df = pd.concat([features_df.reset_index(drop=True), details_df.reset_index(drop=True)], axis=1, ignore_index=True)
    output_df.columns = list(features_df.columns) + list(details_df.columns)
    output_df["score"] = scores
    output_df["classification"] = predictions
    return output_df


def plot_elevation_profiles(profiles: List[dict]):
    fig = go.Figure()
    for i, profile in enumerate(profiles):
        if outliers_df["score"].iloc[i] > 0:
            color = "green"
            opacity = 0.5
        else:
            color = "red"
            opacity = 1.0
        fig.add_scatter(x=profile["d_vec"], y=profile["elev_vec"], opacity=opacity, marker={"color": color}, hovertemplate=f"{profile['witness']}")
    fig.update_layout(
        xaxis_title="Distance, km",
        yaxis_title="Elevation, m"
    )
    return fig


def plot_distance_vs_rssi(outliers_df: pd.DataFrame):
    fig = px.scatter(outliers_df, x="distance_m", y="rssi", color="classification", hover_data=["_to"])
    fig.update_xaxes(range=[0, 50e3])
    fig.update_yaxes(range=[-140, -50])
    return fig


def monte_carlo_trilateration(X: pd.DataFrame, witness_coords: list, model, hotspot_dict):
    mode = model.steps[1][0]
    if mode not in ["svr", "gaussianprocessregressor"]:
        raise TypeError(f"Unknown model type: {model.steps[1][0]}")

    X_eval = np.array(X.drop(["distance_m"], axis=1))

    if mode == "svr":
        eval_mean = model.predict(X_eval)
    elif mode == "gaussianprocessregressor":
        eval_mean, eval_std = model.predict(X_eval, return_std=True)

    N = 1000

    asserted_location = (hotspot_dict["latitude"], hotspot_dict["longitude"])
    asserted_hex_res8 = h3.geo_to_h3(asserted_location[0], asserted_location[1], 8)
    predicted_locations = []
    for i in range(N):
        idx1, idx2, idx3 = np.random.permutation(X_eval.shape[0])[:3]
        u = haversine(witness_coords[idx1], witness_coords[idx2], unit=Unit.KILOMETERS)
        if mode == "svr":
            r1 = eval_mean[idx1]
            r2 = eval_mean[idx2]
        elif mode == "gaussianprocessregressor":
            r1 = np.random.normal(eval_mean[idx1], eval_std[idx1], 1)
            r2 = np.random.normal(eval_mean[idx2], eval_std[idx2], 1)
        else:
            raise Exception("Unknown mode")

        if r1 > 50 or r2 > 50:
            # try again
            i -= 1
            continue

        x_prime = (r1**2 - r2**2 + u**2) / (2*u)
        if (r1**2 - x_prime**2) < 0:
            # chance that sampled radii are negative, especially when extrapolating gaussian process model
            continue
        y_prime_1 = np.sqrt(r1**2 - x_prime**2)
        y_prime_2 = -y_prime_1

        phi_1 = get_bearing(witness_coords[idx1][0], witness_coords[idx1][1], y_prime_1 + witness_coords[idx1][0], x_prime + witness_coords[idx1][1])
        phi_2 = get_bearing(witness_coords[idx1][0], witness_coords[idx1][1], y_prime_2 + witness_coords[idx1][0], x_prime + witness_coords[idx1][1])

        pt_1 = inverse_haversine(witness_coords[idx1], r1, phi_1)
        pt_2 = inverse_haversine(witness_coords[idx1], r1, phi_2)

        if haversine(witness_coords[idx3], pt_1) < haversine(witness_coords[idx3], pt_2):
            predicted_location = pt_1
        else:
            predicted_location = pt_2

        predicted_locations.append(predicted_location)

    predicted_lat = [c[0] for c in predicted_locations]
    predicted_lon = [c[1] for c in predicted_locations]
    monte_carlo_results = pd.DataFrame([predicted_lat, predicted_lon]).transpose()
    monte_carlo_results.columns = ["lat", "lon"]

    fig = px.density_mapbox(monte_carlo_results, lat="lat", lon="lon", zoom=8, radius=10)
    # fig = px.scatter_mapbox(monte_carlo_results, lat="lat", lon="lon", zoom=9)
    fig.update_layout(mapbox_style="dark",
                      mapbox_accesstoken=open(".mapbox_token").read(),
                      showlegend=False,
                      margin={'l':0, 'r':0, 'b':0, 't':0})
    fig.update(layout_coloraxis_showscale=False)
    return fig, monte_carlo_results


def probability_by_hex_resolution(monte_carlo_results: pd.DataFrame, hotspot_dict: dict, k: int):
    asserted_hex = h3.geo_to_h3(hotspot_dict["latitude"], hotspot_dict["longitude"], 8)
    rings = h3.k_ring(asserted_hex, k)
    n_points = monte_carlo_results.shape[0]
    pts_in_hex = 0
    for i in range(n_points):
        if h3.geo_to_h3(monte_carlo_results.iloc[i].lat, monte_carlo_results.iloc[i].lon, 8) in rings:
            pts_in_hex += 1
    try:
        p = str(np.round(100 * pts_in_hex / n_points, 1))
    except ZeroDivisionError:
        p = "0"
    return p, [h3.h3_to_geo_boundary(h) for h in rings]


st.title("Helium Topographical Analysis")
# hotspot_address = "11PJ5fKGmL3or49Kers5ercC9DfYr2BQMeYE3QCbaogvf25e91h"
hotspot_address = st.text_input("Hotspot Address", placeholder="11PJ5fKGmL3or49Kers5ercC9DfYr2BQMeYE3QCbaogvf25e91h")

model_type = st.radio("Select Regression Model Type", ["SVM", "Gaussian Process"])
k = st.slider("Number of res8 KRings for location verification", min_value=1, max_value=9, step=1, value=5)
run_button = st.button("Run Simulation")
if run_button:
    with st.spinner("Running Monte Carlo Simulation..."):
        try:
            hotspot_dict = get_hotspot_dict(db, hotspot_address)
            features_df, details_df, witness_coords, profiles = generate_features(db, hotspot_address)
            outliers_df = find_outliers(features_df, details_df, iso_forest)

            if model_type == "SVM":
                fig, results = monte_carlo_trilateration(features_df, witness_coords, svm, hotspot_dict)
            elif model_type == "Gaussian Process":
                fig, results = monte_carlo_trilateration(features_df, witness_coords, gp, hotspot_dict)
            else:
                raise ValueError("Unknown Model Type.")

            p, polygons = probability_by_hex_resolution(results, hotspot_dict, k)

            for polygon in polygons:
                hex_lat = [c[0] for c in polygon]
                hex_lon = [c[1] for c in polygon]
                fig.add_scattermapbox(lat=hex_lat, lon=hex_lon, fill="toself", marker={"size": 0, "color": "red"})
            fig.add_scattermapbox(lat=[np.mean(results["lat"])], lon=[np.mean(results["lon"])],
                                  hovertemplate="Predicted Location", marker={"size": 20})
            st.subheader("Trilateration Results")
            st.plotly_chart(fig)
            st.metric(f"Percentage of trilateration predictions that lie within {k} kRings from asserted res8 hex: ", value=f"{p}%")

            st.subheader("Elevation Profiles")

            st.plotly_chart(plot_elevation_profiles(profiles))

            st.subheader("Gaming Analysis")
            st.metric(f"Number of Outlier Receipts / Total Witnesses", value=f"{np.sum(outliers_df['score'] < 0)} / {outliers_df.shape[0]}")
            st.dataframe(outliers_df)

            st.subheader("Distance vs. RSSI")
            st.plotly_chart(plot_distance_vs_rssi(outliers_df))

        except ValueError or TypeError:
            st.error("Error processing simulation. This likely means that we have not loaded any witness data for this hotspot yet.")

