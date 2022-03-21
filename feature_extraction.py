from surface_roughness import *
from typing import List
from haversine import haversine
from haversine import Direction
import h3
from rasterio.windows import Window


def get_bearing(lat1, lon1, lat2, lon2):
    dLon = (lon2 - lon1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x,y)

    return brng


def generate_profile_from_path(path: dict, dataset: DatasetReader, elevation_map: np.array, window: Window):
    (lat1, lon1) = (path["coords_beacon"][1], path["coords_beacon"][0])
    (lat2, lon2) = (path["coords_witness"][1], path["coords_witness"][0])
    bearing = get_bearing(lat1, lon1, lat2, lon2)
    d_vec, elev_vec = get_profile(dataset, elevation_map, window, lat1, lon1, path["distance_m"] / 1000, bearing)
    elev_vec[0] += path["elevation_beacon"]
    elev_vec[-1] += path["elevation_witness"]
    return d_vec, elev_vec


def extract_topographic_features(d_vec: np.array, elev_vec: np.array):
    elev_adj = level_profile(d_vec, elev_vec)
    features = {
        "ra": ra_roughness(elev_adj),
        "rq": rq_roughness(elev_adj),
        "rp": rp_roughness(elev_adj),
        "rv": rv_roughness(elev_adj),
        "rz": rz_roughness(elev_adj),
        "rsk": rsk_roughness(elev_adj),
        "rku": rku_roughness(elev_adj),
        "deepest_barrier": deepest_barrier(elev_adj),
        "n_barriers": n_barriers(elev_adj)
    }
    return features


def process_witness_paths(witness_paths: List[dict], dataset: DatasetReader, elevation_map: np.array, window: Window, return_coords: bool = False):
    path_features, path_details, witness_coords, profiles = [], [], [], []
    for path in witness_paths:
        if not path["tx_power"] or not path["distance_m"]:
            continue
        if path["distance_m"] > 50e3:
            continue
        try:
            d_vec, elev_vec = generate_profile_from_path(path, dataset, elevation_map, window)
        except IndexError:
            # print(f"Witness path out of range: {path['_from']} to {path['_to']}, distance: {path['distance_m']}")
            continue

        # get topographic features
        try:
            features = extract_topographic_features(d_vec, elev_vec)
        except (np.linalg.LinAlgError, SystemError):
            # hotspots are too close to create elevation profile
            continue

        # add signal-specific features
        features["tx_power"] = path["tx_power"]
        features["gain_beacon"] = path["gain_beacon"]
        features["gain_witness"] = path["gain_witness"]
        features["rssi"] = path["rssi"]
        features["snr"] = path["snr"]

        # add distance
        features["distance_m"] = path["distance_m"]
        path_features.append(features)
        path_details.append({"_from": path["_from"], "_to": path["_to"], "witness_owner": path["witness_owner"]})
        if return_coords:
            # coords = path["coords_witness"]
            coords = path["coords_beacon"]
            witness_coords.append(tuple([coords[1], coords[0]]))
            profiles.append({"d_vec": d_vec, "elev_vec": elev_vec, "witness": path["_to"], "beacon": path["_from"]})

    if return_coords:
        return path_features, path_details, witness_coords, profiles
    else:
        return path_features, path_details
