import pandas as pd
from pyArango.theExceptions import AQLFetchError

from app import generate_features, find_outliers
import pickle
from pyArango.connection import Connection, Database
import os
from arango_queries import list_hotspots_near_coords
import matplotlib.pyplot as plt


TRAINED_ISO_PATH = "static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl"
denylist = pd.read_csv("denylist.csv", index_col=None, header=None).drop(1, axis=1)
denylist.columns = ["address"]

nominal = pd.read_csv("known-good.csv", index_col=None, header=None)
nominal.columns = ["address"]


def load_model(path: str):
    print("Loading trained model...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("done.")
    return model


try:
    c = Connection(
        arangoURL=os.getenv('ARANGO_URL'),
        username=os.getenv('ARANGO_USERNAME'),
        password=os.getenv('ARANGO_PASSWORD')
    )
except ConnectionError:
    raise Exception('Unable to connect to the ArangoDB instance. Please check that it is running and that you have supplied the correct URL/credentials in the .env file.')
db: Database = c['helium-graphs']


iso_model = load_model(TRAINED_ISO_PATH)


# GAMING LIKELIHOOD
gaming_results = []
for i, hotspot in enumerate(denylist["address"]):
    if i % 100 == 0:
        print(i)
    try:
        features_df, details_df, _, _ = generate_features(db, hotspot, "inbound")
        output_df = find_outliers(features_df, details_df, iso_model)
        gaming_results.append([len(output_df), len(output_df[output_df["classification"] < 0])])
    except (AQLFetchError, ValueError):
        continue

gaming_results_df = pd.DataFrame(gaming_results)
gaming_results_df.columns = ["n_receipts", "n_anomalies"]
gaming_results_df["percent"] = gaming_results_df["n_anomalies"] / gaming_results_df["n_receipts"]

# NOMINAL LIKELIHOOD

nominal_results = []
for i, hotspot in enumerate(nominal["address"]):
    if i % 100 == 0:
        print(i)
    try:
        features_df, details_df, _, _ = generate_features(db, hotspot, "inbound")
        output_df = find_outliers(features_df, details_df, iso_model)
        nominal_results.append([len(output_df), len(output_df[output_df["classification"] < 0])])
    except (AQLFetchError, ValueError):
        continue

nominal_results_df = pd.DataFrame(nominal_results)
nominal_results_df.columns = ["n_receipts", "n_anomalies"]
nominal_results_df["percent"] = nominal_results_df["n_anomalies"] / nominal_results_df["n_receipts"]
