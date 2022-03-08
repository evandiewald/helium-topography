import rasterio
from gis_utils import *
from arango_queries import *
from feature_extraction import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from pyArango.connection import Connection, Database
from pyArango.theExceptions import AQLFetchError
import pandas as pd
import os
import pickle
import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt


HOTSPOTS_PER_LOCATION = 100
WITNESSES_PER_HOTSPOT = 100
pd.set_option('display.max_colwidth', None)

load_dotenv()

try:
    c = Connection(
        arangoURL=os.getenv("ARANGO_URL"),
        username=os.getenv("ARANGO_USERNAME"),
        password=os.getenv("ARANGO_PASSWORD")
    )
except ConnectionError:
    raise Exception("Unable to connect to the ArangoDB instance. Please check that it is running and that you have supplied the correct URL/credentials in the .env file.")
db: Database = c["helium-graphs"]

city_list = [
    {"city": "San Francisco", "coordinates": [37.7749, -122.4194]},
    {"city": "Antwerp", "coordinates": [51.2213, 4.4051]},
    {"city": "Utrecht", "coordinates": [52.0907, 5.1214]},
    {"city": "Denver", "coordinates": [39.7392, -104.9903]},
    {"city": "Dallas", "coordinates": [32.7767, -96.7970]},
    {"city": "Singapore", "coordinates": [1.3521, 103.8198]},
    {"city": "Phoenix", "coordinates": [33.4484, -112.0740]},
    {"city": "Beijing", "coordinates": [39.9042, 116.4074]},
    {"city": "Nanning, CN", "coordinates": [22.8167, 108.3669]},
    {"city": "Izombe, Nigeria", "coordinates": [5.6344, 6.8592]},
    {"city": "Lisbon", "coordinates": [38.7223, -9.1393]},
    {"city": "Rio de Janeiro", "coordinates": [-22.9068, -43.1729]},
    {"city": "Sydney", "coordinates": [-33.8688, 151.2093]},
    {"city": "Mexico City", "coordinates": [19.4326, -99.1332]},
    {"city": "Moscow", "coordinates": [55.7558, 37.6173]}
]


path_features, path_details = [], []
with rasterio.open(os.getenv("VRT_PATH")) as dataset:
    for city in city_list:
        print(f"Processing hotspots from {city['city']}")
        query_coords = city["coordinates"]
        hotspots_list = list_hotspots_near_coords(db, query_coords, limit=HOTSPOTS_PER_LOCATION, search_radius_m=50000)
        elevation_map, window = get_local_elevation_map(dataset, query_coords[0], query_coords[1], range_km=250)
        elevation_map[elevation_map == dataset.nodata] = 0
        for hotspot in hotspots_list:
            try:
                witness_paths = get_witnesses_for_hotspot(db, hotspot["address"], WITNESSES_PER_HOTSPOT)
            except AQLFetchError:
                # print(f"No witnesses for hotspot {hotspot['address']}")
                continue
            features, details = process_witness_paths(witness_paths, dataset, elevation_map, window)
            path_features += features
            path_details += details


################################ OUTLIER DETECTION: ISOLATION FOREST ###########################################

X = pd.DataFrame(path_features)
paths_df = pd.DataFrame(path_details)
X, paths_df = shuffle(X, paths_df, random_state=0)

# train Isolation forest
iso_forest = make_pipeline(StandardScaler(), IsolationForest(random_state=0))
y_pred = iso_forest.fit_predict(X)
scores = iso_forest.decision_function(X)

# isolation score < 0 indicates outliers
output_df = pd.concat([X.reset_index(drop=True), paths_df.reset_index(drop=True)], axis=1, ignore_index=True)
output_df.columns = list(X.columns) + list(paths_df.columns)
output_df["score"] = scores

suspicious_witnesses = pd.DataFrame(output_df["_to"][output_df["score"] < 0].value_counts(sort=True, ascending=False))
owners = []
for witness in suspicious_witnesses.index:
    owners.append(paths_df["witness_owner"][paths_df["_to"] == witness].iloc[0])
suspicious_witnesses["owner"] = owners


################################ REGRESSION: SVM + GAUSSIAN PROCESS ###########################################

# random 80/20 train/test split
y = np.array(X["distance_m"]) / 1000
X2_df = X.drop(["distance_m"], axis=1)
X2 = np.array(X2_df)
ratio_training = 0.8
X2, y = shuffle(X2, y, random_state=0)
X_train = X2[:int(X2.shape[0]*ratio_training),:]
y_train = y[:int(X2.shape[0]*ratio_training)]
X_test = X2[int(X2.shape[0]*ratio_training):,:]
y_test = y[int(X2.shape[0]*ratio_training):]

# train gaussian process regressor
regr_gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(random_state=0, copy_X_train=False))
regr_gp.fit(X_train, y_train)
pred_mean_gp, pred_std_gp = regr_gp.predict(X_test, return_std=True)

# train SVM with linear kernel
regr_svm = make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0, cache_size=1000, gamma="scale"))
regr_svm.fit(X_train, y_train)
pred_svm = regr_svm.predict(X_test)

# PLOTTING
plt.plot(y_test, y_test, "k-", label="y_pred = y_test")
plt.scatter(y_test, pred_mean_gp, alpha=0.2, label="Gaussian Process")
plt.scatter(y_test, pred_svm, alpha=0.2, label="SVM")
plt.legend()
plt.ylabel("Predicted Distance, km")
plt.xlabel("Actual Distance, km")
plt.xlim(0,50)
plt.ylim(0,50)
plt.show()

# shows the difficulty of only using RSSI values...need to encode some other features
plt.scatter(X["distance_m"] / 1000, X["rssi"])
plt.ylim(-140, -50)
plt.xlabel("Distance, km")
plt.ylabel("RSSI")
plt.show()

feature_names = X2_df.columns
pd.Series(abs(regr_svm.steps[1][1].coef_[0]), index=feature_names).plot(kind='barh')
plt.subplots_adjust(left=0.25)
plt.ylabel("Feature")
plt.xlabel("Relative Importance")
plt.show()

diff_svm = np.abs(y_test - pred_svm)
diff_gp = np.abs(y_test - pred_mean_gp)
error = np.linspace(0, 25)
cdf_svm = np.zeros_like(error)
cdf_gp = np.zeros_like(error)
for i in range(len(error)):
    cdf_svm[i] = np.sum(diff_svm < error[i]) / len(y_test)
    cdf_gp[i] = np.sum(diff_gp < error[i]) / len(y_test)
plt.plot(error, cdf_gp, label="GP")
plt.plot(error, cdf_svm, label="SVM")
plt.xlabel("Absolute Error, km")
plt.ylabel("CDF")
plt.legend()
plt.show()


# SAVING TRAINED MODELS
with open(f"static/trained_models/gaussian_process/{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.mdl", "wb") as f:
    pickle.dump(regr_gp, f, pickle.HIGHEST_PROTOCOL)

with open(f"static/trained_models/svm/{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.mdl", "wb") as f:
    pickle.dump(regr_svm, f, pickle.HIGHEST_PROTOCOL)

with open(f"static/trained_models/isolation_forest/{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.mdl", "wb") as f:
    pickle.dump(iso_forest, f, pickle.HIGHEST_PROTOCOL)