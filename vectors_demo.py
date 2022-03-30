import math

from arango_queries import get_witnesses_for_hotspot
from pyArango.connection import Connection, Database
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HOTSPOT_ADDRESS = "11RaVmAadLFRstHS5sZZ9Yr52nw3hTPoNaLwERx5Mjfrxa631Wh"
# HOTSPOT_ADDRESS = "11toz227sLe3xoA7vWgGt5ZnMQBDzuMdodYkcWpMFt9ERP5VfQq"
# HOTSPOT_ADDRESS = "112sLpwVU7Kx4pxBDFEAQPpatmELE8UmvXP25muCZQmykv4j2s6r"
# HOTSPOT_ADDRESS = "11RA5Q53xN2RpALQ5aRhi1ztxt1YoZG4CyUXL6t7mwgkHZDj9Av"
N = 3 # num nearest hotspots in vector search

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


witnesses = get_witnesses_for_hotspot(db, HOTSPOT_ADDRESS)
witnesses = [w for w in witnesses if w["distance_m"] < 50000]
witnesses_df = pd.DataFrame(witnesses)
beacon_coords = np.array([w["coords_beacon"] for w in witnesses])


def calc_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def calc_theta(pt1, pt2):
    return math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])


def get_distance_matrices(witnesses):
    distance_matrix = np.zeros((len(witnesses), len(witnesses)))
    theta_matrix = np.zeros_like(distance_matrix)

    for ii in range(len(witnesses)):
        for jj in range(len(witnesses)):
            if ii == jj:
                distance_matrix[ii,jj] = np.nan
                theta_matrix[ii,jj] = np.nan
                continue
            distance_matrix[ii,jj] = calc_distance(witnesses[ii]["coords_beacon"], witnesses[jj]["coords_beacon"])
            theta_matrix[ii,jj] = calc_theta(witnesses[ii]["coords_beacon"], witnesses[jj]["coords_beacon"])
    return distance_matrix, theta_matrix


distance_matrix, theta_matrix = get_distance_matrices(witnesses)
# fig, ax = plt.subplots()
net_dx, net_dy = np.zeros((len(witnesses),)), np.zeros((len(witnesses),))
# ax.scatter(beacon_coords[:,0], beacon_coords[:,1], alpha=0.3)
for i in range(len(witnesses)):
    # ax.annotate(str(i), (beacon_coords[i,0], beacon_coords[i,1]))
    # find nearest N vectors
    nearest_idx = np.argpartition(distance_matrix[i,:], N)[:N]
    theta = theta_matrix[i,nearest_idx]
    r = [(witnesses[idx]["rssi"] - witnesses[i]["rssi"]) / (witnesses[idx]["rssi"] - witnesses[i]["distance_m"]) for idx in nearest_idx]
    dx = [r[j] * math.cos(theta[j]) for j in range(len(theta))]
    dy = [r[j] * math.sin(theta[j]) for j in range(len(theta))]
    # for j in range(len(theta)):
    #     ax.arrow(beacon_coords[i,0], beacon_coords[i,1], dx[j], dy[j],
    #              color="gray", head_width=0.01)
    net_dx[i], net_dy[i] = np.sum(dx), np.sum(dy)
    net_theta = math.atan2(net_dy[i], net_dx[i])
    # ax.arrow(beacon_coords[i,0], beacon_coords[i,1], net_dx[i], net_dy[i],
    #          color="orange", head_width=0.005)

# plt.show()


from scipy.interpolate import griddata
from haversine import haversine

x_steps = int(np.abs(max(beacon_coords[:,0]) - min(beacon_coords[:,0])) * 1000)
y_steps = int(np.abs(max(beacon_coords[:,1]) - min(beacon_coords[:,1])) * 1000)
grid_x, grid_y = np.meshgrid(np.linspace(min(beacon_coords[:,0]), max(beacon_coords[:,0]), x_steps),
                             np.linspace(min(beacon_coords[:,1]), max(beacon_coords[:,1]), y_steps))
grid_z = griddata(beacon_coords, [w["rssi"] for w in witnesses], (grid_x, grid_y), method="linear")
plt.title("RSSI")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.imshow(grid_z, origin="lower")
plt.colorbar()
plt.show()

grid_dx = griddata(beacon_coords, net_dx, (grid_x, grid_y), method="linear")
plt.imshow(grid_dx, origin="lower")
plt.title("dRSSI / dx")
plt.colorbar()
plt.show()

grid_dy = griddata(beacon_coords, net_dy, (grid_x, grid_y), method="linear")
plt.imshow(grid_dy, origin="lower")
plt.title("dRSSI / dy")
plt.colorbar()
plt.show()

H,W = grid_z.shape
grid_div = np.zeros_like(grid_z)

for i in range(1, H - 1):
    for j in range(1, W - 1):
        grid_div[i,j] = grid_dx[i,j+1] - grid_dx[i,j-1] + grid_dy[i+1,j] - grid_dy[i-1,j]

plt.imshow(grid_div, origin="lower")
plt.title("divergence")
plt.colorbar()
plt.show()



plt.pcolor(grid_x, grid_y, grid_div)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

coords_idx_min_div = np.nanargmin(grid_div)
coords_idx_max_rssi = np.nanargmax(grid_z)
predicted_coords_div = (grid_y.flatten()[coords_idx_min_div], grid_x.flatten()[coords_idx_min_div])
predicted_coords_rssi = (grid_y.flatten()[coords_idx_max_rssi], grid_x.flatten()[coords_idx_max_rssi])
asserted_coords = (witnesses[0]['coords_witness'][1], witnesses[0]['coords_witness'][0])
print(f"Predicted coords (divergence): {predicted_coords_div}")
print(f"Distance from assert (divergence): {np.round(haversine(predicted_coords_div, asserted_coords), 3)} km")
print(f"Distance from assert (max rssi): {np.round(haversine(predicted_coords_rssi, asserted_coords), 3)} km")

fig, ax = plt.subplots()
ax.streamplot(grid_x, grid_y, grid_dx, grid_dy, color=np.sqrt(grid_dx**2+grid_dy**2), cmap="autumn")
# ax.quiver(grid_x, grid_y, grid_dx, grid_dy)
ax.scatter(asserted_coords[1], asserted_coords[0], marker="+", color="k", s=100)
ax.set_aspect("equal")
plt.show()




x_steps = int(np.abs(max(beacon_coords[:,0]) - min(beacon_coords[:,0])) * 50)
y_steps = int(np.abs(max(beacon_coords[:,1]) - min(beacon_coords[:,1])) * 50)
grid_x, grid_y = np.meshgrid(np.linspace(min(beacon_coords[:,0]), max(beacon_coords[:,0]), x_steps),
                             np.linspace(min(beacon_coords[:,1]), max(beacon_coords[:,1]), y_steps))
grid_dx = griddata(beacon_coords, net_dx, (grid_x, grid_y), method="linear")
grid_dy = griddata(beacon_coords, net_dy, (grid_x, grid_y), method="linear")

fig, ax = plt.subplots()
# ax.streamplot(grid_x, grid_y, grid_dx, grid_dy, color=np.sqrt(grid_dx**2+grid_dy**2), cmap="autumn")
ax.quiver(grid_x, grid_y, -grid_dx, -grid_dy)
ax.scatter(asserted_coords[1], asserted_coords[0], marker="+", color="k", s=100)
ax.set_aspect("equal")
plt.show()