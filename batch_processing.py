import pandas as pd
from pyArango.theExceptions import AQLFetchError

from app import generate_features, find_outliers, monte_carlo_trilateration, get_hotspot_dict
import pickle
from pyArango.connection import Connection, Database
import os


TRAINED_ISO_PATH = "static/trained_models/isolation_forest/2022-02-04T16_31_09.mdl"
TRAINED_SVM_PATH = "static/trained_models/svm/2022-02-06T16_23_54.mdl"
GATEWAY_INVENTORY_PATH = "G:/Downloads/gateway_inventory_01289021.csv.gz"


gateway_inventory = pd.read_csv(GATEWAY_INVENTORY_PATH, compression="gzip")


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
svm_model = load_model(TRAINED_SVM_PATH)


results = []
for i, hotspot in enumerate(gateway_inventory["address"]):
    if i % 100 == 0:
        print(i)
    try:
        hotspot_dict = get_hotspot_dict(db, hotspot)
        features_df, details_df, witness_coords, _ = generate_features(db, hotspot, "inbound")
        output_df = find_outliers(features_df, details_df, iso_model)

        _, _, _, prediction_error = monte_carlo_trilateration(features_df, witness_coords, svm_model, hotspot_dict, 1)
        results.append({
            "address": hotspot,
            "n_receipts": len(output_df),
            "n_outliers": len(output_df[output_df["classification"] < 0]),
            "prediction_error": prediction_error
        })
    except (AQLFetchError, ValueError):
        continue

results_df = pd.DataFrame(results)
results_df["percent"] = results_df["n_anomalies"] / results_df["n_receipts"]


