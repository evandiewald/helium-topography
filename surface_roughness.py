import numpy as np
from scipy.stats import skew, kurtosis
from itertools import groupby
from rasterio import DatasetReader
from rasterio.windows import Window
from haversine import inverse_haversine


"""https://en.wikipedia.org/wiki/Surface_roughness"""


def get_elevation_index(dataset: DatasetReader, lat: float, lon: float):
    row, col = dataset.index(lon, lat)
    return row, col


def get_profile(dataset: DatasetReader, data: np.array, window: Window, lat1, lon1, distance, bearing):
    """step in increments of 30m according to resolution of topography data"""
    d_vec = np.arange(0, distance, 0.03)
    elev_vec = np.zeros_like(d_vec)

    for j in range(len(d_vec)):
        (lat, lon) = inverse_haversine((lat1, lon1), d_vec[j], bearing)
        (row, col) = get_elevation_index(dataset, lat, lon)
        map_row, map_col = int(row - window.row_off), int(col - window.col_off)
        if map_row < 0 or map_col < 0:
            raise IndexError("Witness path not captured by local elevation map")
        elev_vec[j] = data[map_row, map_col]

    return d_vec, elev_vec


def level_profile(x, y):
    """fit based on transmitter -> receiver"""
    p = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
    y_fit = np.polyval(p, x)
    out = y - y_fit
    # fix near-zero values
    out[0] = 0
    out[-1] = 0
    return out


def ra_roughness(y_adj):
    """profile mean deviation"""
    return np.mean(np.abs(y_adj))


def rq_roughness(y_adj):
    """root mean squared"""
    return np.sqrt(np.mean(np.dot(y_adj, y_adj)))


def rp_roughness(y_adj):
    """highest peak"""
    return np.max(y_adj)


def rv_roughness(y_adj):
    """lowest valley"""
    return np.min(y_adj)


def rz_roughness(y_adj):
    """peak to valley height"""
    return rp_roughness(y_adj) - rv_roughness(y_adj)


def rsk_roughness(y_adj):
    """skewness"""
    return skew(y_adj)


def rku_roughness(y_adj):
    """kurtosis"""
    return kurtosis(y_adj)


def len_iter(items):
    return sum(1 for _ in items)


def deepest_barrier(y_adj):
    try:
        return max(len_iter(run) for val, run in groupby(y_adj > 0) if val)
    except ValueError:
        return 0


def n_barriers(y_adj):
    return len([sum(1 for i in g) for k,g in groupby(y_adj > 0)])


def is_in_line_of_sight(y_adj) -> bool:
    return False if n_barriers(y_adj) > 0 else True