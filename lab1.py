from pyorbital.orbital import Orbital
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

orb = Orbital("NOAA-19")
start = datetime.now()
obs_lat, obs_long, obs_alt = 55.75, 37.62, 0.2

time_array = orb.get_next_passes(start, 24, obs_long, obs_lat, obs_alt)


arr = []
for i in range(len(time_array)):
    azimuth = []
    elevation = []
    az, el = orb.get_observer_look(time_array[i][0], obs_long, obs_lat, obs_alt)
    azimuth.append(az)
    elevation.append(el)
    az, el = orb.get_observer_look(time_array[i][2], obs_long, obs_lat, obs_alt)
    azimuth.append(az)
    elevation.append(el)
    az, el = orb.get_observer_look(time_array[i][1], obs_long, obs_lat, obs_alt)
    azimuth.append(az)
    elevation.append(el)
    arr.append([azimuth, elevation])

print(arr)

print(azimuth)
print(elevation)


fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(projection='polar')
ax1.set_theta_zero_location('N')
ax1.set_rlim(bottom=90, top=0)
for azimuth, elevation in arr:
    plt.plot(np.deg2rad(azimuth), elevation, color='black')
ax1.grid(True)
plt.show()
