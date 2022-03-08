from rasterio import DatasetReader
from rasterio.windows import Window
import numpy as np
from typing import Tuple


def get_local_elevation_map(dataset: DatasetReader, lat: float, lon: float, range_km: int) -> Tuple[np.array, Window]:
    """
    Pulls the local elevation map near query coordinates as a numpy array, as well as the offsets relative to the global map.
    :param dataset: The open rasterio.DatasetReader
    :param lat: query latitude
    :param lon: query longitude
    :param range_km: size of the window in both dimensions (km)
    :return: the elevation map array and the Window object
    """
    km_per_degree = 111.3
    (window_height, window_width) = (int(range_km / (dataset.res[0] * km_per_degree)), int(range_km / (dataset.res[1] * km_per_degree)))
    row_offset, col_offset = dataset.index(lon, lat)
    col_offset -= window_width / 2
    row_offset -= window_height / 2
    window = Window(col_offset, row_offset, window_width, window_height)
    return dataset.read(1, window=window), window




