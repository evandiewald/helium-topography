import h3.unstable.vect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from typing import List, Literal

import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv

from sqlalchemy.engine import Engine, create_engine

import os
import rasterio

from sql_queries import get_hotspot_dict_sql, get_witnesses_for_hotspot_sql, get_witnesses_of_hotspot_sql

from gis_utils import get_local_elevation_map
from feature_extraction import process_witness_paths, get_bearing
import numpy as np
from haversine import haversine, inverse_haversine, Unit



TRAINED_SVM_PATH = "static/trained_models/svm/2022-02-06T16_23_54.mdl"
TRAINED_GP_PATH = "static/trained_models/gaussian_process/2022-02-04T16_28_14.mdl"
TRAINED_ISO_PATH = "static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl"
GAMING_DISTRIBUTION_PATH = "static/assets/gaming_results.csv"
NOMINAL_DISTRIBUTION_PATH = "static/assets/nominal_results.csv"


@st.experimental_singleton
def load_model(path: str):
    print("Loading trained model...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("done.")
    return model


@st.experimental_singleton
def load_csv(path: str):
    return pd.read_csv(path)


svm = load_model(TRAINED_SVM_PATH)
# gp = load_model(TRAINED_GP_PATH)
iso_forest = load_model(TRAINED_ISO_PATH)
gaming_dist = load_csv(GAMING_DISTRIBUTION_PATH)
nominal_dist = load_csv(NOMINAL_DISTRIBUTION_PATH)

# put db credentials in .env file
load_dotenv()

try:
    engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
except ConnectionError:
    raise Exception('Unable to connect to the Postgres db. Please check that it is running and that you have supplied the correct URL/credentials in the .env file.')


# EVALUATION
def generate_features(engine: Engine, hotspot_address: str, witness_direction: Literal["inbound", "outbound"] = "inbound"):
    hotspot_dict = get_hotspot_dict_sql(engine, hotspot_address)

    if witness_direction == "inbound":
        witness_paths = get_witnesses_for_hotspot_sql(engine, hotspot_address, limit=1000)
    else:
        witness_paths = get_witnesses_of_hotspot_sql(engine, hotspot_address, limit=1000)

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


def monte_carlo_trilateration(X: pd.DataFrame, witness_coords: list, model, hotspot_dict, k, show_radii=False,
                              show_predictions=True, show_asserted_location=True):
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

    # ADDED FOR TEMPORARY VISUALIZATION
    circles = []
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

        # if haversine(witness_coords[idx3], pt_1) < haversine(witness_coords[idx3], pt_2):
        if haversine(asserted_location, pt_1) < haversine(asserted_location, pt_2):
            predicted_location = pt_1
        else:
            predicted_location = pt_2

        predicted_locations.append(predicted_location)

        ##TEMP
        # lat, lon, radius
        circles.append([witness_coords[idx1][0], witness_coords[idx1][1], r1*1000])
        # circles.append([witness_coords[idx2][0], witness_coords[idx2][1], r2*1000])
    predicted_lat = [c[1] for c in predicted_locations]
    predicted_lon = [c[0] for c in predicted_locations]
    monte_carlo_results = pd.DataFrame([predicted_lat, predicted_lon]).transpose()
    monte_carlo_results.columns = ["lat", "lon"]

    circles_df = pd.DataFrame(circles)
    circles_df.columns = ["lat", "lon", "radius"]
    # fig = px.density_mapbox(monte_carlo_results, lat="lat", lon="lon", zoom=8, radius=10)
    # # fig = px.scatter_mapbox(monte_carlo_results, lat="lat", lon="lon", zoom=9)
    # fig.update_layout(mapbox_style="dark",
    #                   mapbox_accesstoken=os.getenv("MAPBOX_API_KEY"),
    #                   showlegend=False,
    #                   margin={'l':0, 'r':0, 'b':0, 't':0})
    # fig.update(layout_coloraxis_showscale=False)
    rings = pd.DataFrame(h3.k_ring(asserted_hex_res8, k))
    rings.columns = ["hex"]

    print(monte_carlo_results)

    fig = pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(
            latitude=monte_carlo_results["lat"].mean(),
            longitude=monte_carlo_results["lon"].mean(),
            zoom=10,
            pitch=0,
        ),
        layers=[],
    )

    if show_radii:
        fig.layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=circles_df,
                get_position='[lon, lat]',
                get_radius="radius",
                filled=False,
                stroked=True,
                get_line_color=[0, 255, 0, 10],
                get_line_width=100,
                get_fill_color=[0, 0, 0, 10]
            )
        )

    if show_predictions:
        # fig.layers.append(
            # pdk.Layer(
            #     'HexagonLayer',
            #     data=monte_carlo_results,
            #     get_position='[lon, lat]',
            #     radius=1206,
            #     elevation_scale=4,
            #     elevation_range=[0, 1000],
            #     # get_fill_color=[180, 0, 200, 140],
            #     pickable=True,
            #     extruded=False,
            # )
        fig.layers.append(
            pdk.Layer(
                'HeatmapLayer',
                data=monte_carlo_results,
                get_position='[lon, lat]',
                get_radius=200,
                get_fill_color=[255, 255, 255, 140],  # Set an RGBA value for fill
                pickable=True
            )
        )
        best_guess = [{"lon": np.median(monte_carlo_results["lon"]),
                       "lat": np.median(monte_carlo_results["lat"]),
                       "icon_data": {
                           "url": "https://cdn-icons-png.flaticon.com/512/59/59325.png",
                           "width": 512,
                           "height": 512,
                           "anchorY": 512
                       }}]

        fig.layers.append(
            pdk.Layer(
                'IconLayer',
                data=best_guess,
                get_position='[lon, lat]',
                get_icon='icon_data',
                get_size=4,
                size_scale=15,
                # get_radius=500,
                # get_fill_color=[0,0, 0],
                pickable=True,
                # filled=True
            )
        )

    if show_asserted_location:
        fig.layers.append(
            pdk.Layer(
                'H3HexagonLayer',
                data=rings,
                get_hexagon="hex",
                get_fill_color=[180, 0, 200, 140],
                extruded=False
            )
        )




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


def bayesian_inference(gaming_dist, nominal_dist, outlier_df, prior: float = 0.1) -> float:
    """
    Use bayesian stats to infer the likelihood of a gamer based on percentage of anomalous paths.
    :param gaming_dist:
    :param nominal_dist:
    :param outlier_df:
    :param prior: The assumed ratio of overall gaming on the network
    :return:
    """
    # find percentage of normal receipts (evidence)
    E = len(outlier_df[outlier_df["classification"] > 0]) / len(outlier_df)
    # likelihood of evidence given gaming, P(E|G) (within 10%)
    pe_g = len(gaming_dist[((1 - gaming_dist["percent"]) < E)]) / len(gaming_dist)
    # likelihood of evidence given nominal, P(E|~G) (within 10%)
    pe_n = len(nominal_dist[((1 - nominal_dist["percent"]) < E)]) / len(nominal_dist)
    # solve for posterior
    # pg_e = 1 / (1 + ((1 / prior) - 1) * (pe_n / pe_g))
    pg_e = pe_g * prior / pe_n
    return pg_e



st.title("Helium Topographical Analysis :balloon:")

with st.expander("About"):
    st.markdown("This research tool uses topographic datasets, machine learning, and a graph databases to help identify gaming activity on the "
                "Helium Network. To start, input a hotspot's b58 hash and click `Run Simulation`. Read more about the methodologies "
                "[here](https://neighborly-beret-bf5.notion.site/Modeling-RF-Propagation-on-the-Helium-Network-with-Topographic-Data-09ad0302eaea46a0a078bbac5f24c2a0).")
    st.markdown("[Github](https://github.com/evandiewald/helium-topography)")
# hotspot_address = "11PJ5fKGmL3or49Kers5ercC9DfYr2BQMeYE3QCbaogvf25e91h"
hotspot_address = st.text_input("Hotspot Address", placeholder="11PJ5fKGmL3or49Kers5ercC9DfYr2BQMeYE3QCbaogvf25e91h")

# model_type = st.radio("Select Regression Model Type", ["SVM", "Gaussian Process"])
model_type = "SVM"
witness_direction: str = st.radio("Select Witness Direction", ["Inbound", "Outbound"])
show_radii = st.checkbox("Show Trilateration Radii", value=False)
show_predictions = st.checkbox("Show Trilateration Heatmap", value=True)
show_asserted_location = st.checkbox("Show Asserted Location", value=True)
k = st.slider("Number of res8 KRings for location verification", min_value=1, max_value=9, step=1, value=5)
run_button = st.button("Run Simulation")
if run_button:
    with st.spinner("Running Monte Carlo Simulation..."):
        # try:
        hotspot_dict = get_hotspot_dict_sql(engine, hotspot_address)
        features_df, details_df, witness_coords, profiles = generate_features(engine, hotspot_address, witness_direction.lower())
        print(features_df)
        outliers_df = find_outliers(features_df, details_df, iso_forest)

        if model_type == "SVM":
            fig, results = monte_carlo_trilateration(features_df, witness_coords, svm, hotspot_dict, k, show_radii,
                                                     show_predictions, show_asserted_location)
        elif model_type == "Gaussian Process":
            # fig, results = monte_carlo_trilateration(features_df, witness_coords, gp, hotspot_dict, k, show_radii, show_predictions)
            pass
        else:
            raise ValueError("Unknown Model Type.")

        p, polygons = probability_by_hex_resolution(results, hotspot_dict, k)

        st.subheader("Trilateration Results")

        with st.expander("About this Chart"):
            st.markdown("[**Multi-trilateration**](https://en.wikipedia.org/wiki/True-range_multilateration)"
                        " is a mathematical technique for geo locating a point given its known distances from two other points."
                        " In this implementation, we use topographic and signal-quality features to predict a hotspot's true location based on its "
                        "witness data. Specifically, these features are fed into a trained machine learning model to predict distances from "
                        "beaconer to witness before solving the trilateration problem in a monte carlo simulation. The more witness data we have,"
                        "the better the prediction.")
            st.markdown("In these charts, we plot a heatmap of those predicted locations and a crosshair icon signifying the median prediction (i.e. our \"best guess\")."
                        " Compare these results to the hotspot's asserted location (pink hexes). The more spread and higher error you see in the trilateration solutions,"
                        " the more likely a hotspot is spoofing their location or witness activity.")
            st.image("static/assets/spoofer-ex.png", caption="A likely spoofer with a broad confidence interval in the trilateration solutions.")
            st.image("static/assets/known-good-ex.png", caption="A nominal hotspot, where most solutions lie near the asserted location.")

        # st.plotly_chart(fig)
        st.pydeck_chart(fig)
        st.metric(f"Percentage of trilateration predictions that lie within {k} kRings from asserted res8 hex: ", value=f"{p}%")

        st.subheader("Elevation Profiles")

        with st.expander("About this Chart"):
            st.markdown("This plot shows the line-of-sight profiles between beaconer and witness for this hotspot. Imagine placing a rope on the "
                        "ground between the two radios and tracking elevation vs. distance. We use an outlier detection algorithm to identify"
                        "anomalous witness receipts, which have dubious signal quality given their line-of-sight characteristics. In this chart, "
                        "outliers show up as red traces, and nominal profiles are green. You can isolate individual traces by double clicking"
                        "the legend.")

        st.plotly_chart(plot_elevation_profiles(profiles))

        st.subheader("Gaming Analysis")

        with st.expander("About this Dataframe"):
            st.markdown("This dataframe shows all the [topographic](https://en.wikipedia.org/wiki/Surface_roughness)"
                        " and signal quality data that we use in our models. Each row represents an "
                        "individual witness pair for this hotspot. The `score` column corresponds to how anomalous that witness path is "
                        "according to our outlier detection algorithm (more negative -> higher likelihood that it is an outlier). "
                        "You can sort any column by clicking the header. The indices match "
                        "the trace identifiers in the elevation profiles chart above.")

        st.metric(f"Number of Outlier Receipts / Total Witnesses", value=f"{np.sum(outliers_df['score'] < 0)} / {outliers_df.shape[0]}")

        posterior = bayesian_inference(gaming_dist, nominal_dist, outliers_df, prior=0.1)
        # st.metric("Gaming Probability", value=f"{np.round(posterior * 100, 1)} %")

        st.dataframe(outliers_df)

        st.subheader("Distance vs. RSSI")

        with st.expander("About this Chart"):
            st.markdown("The [Free Space Path Loss](https://www.pasternack.com/t-calculator-fspl.aspx) equation defines a theoretical "
                        "relationship between RF signal strength (RSSI) and the propagation distance. FSPL assumes an unobstructed line of "
                        "sight between transmitter and receiver, which can be problematic (hence the inclusion of topographic data in this"
                        "analysis). However, gamers may use packet forwarding software to alter the signal characteristics of received"
                        "packets, which should be revealed in this chart. Individual points are colored according to the outlier detection"
                        "algorithm described above.")
            st.image("static/assets/spoofer-rssi.png", caption="Evidence of packet forwarding software - hard RSSI floor at -140 dB.")

        st.plotly_chart(plot_distance_vs_rssi(outliers_df))

        # except:
        #     st.error("Error processing simulation. This likely means that we have not loaded any valid witness data for this hotspot yet. "
        #              "The model only considers individual transmit paths under 100km.")

