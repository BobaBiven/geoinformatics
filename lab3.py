import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from pyorbital.orbital import Orbital
import cv2
import datetime
from scipy import interpolate
import shapefile
from math import radians
from pyproj import Transformer

fs, signal = wavfile.read("/home/askar/Загрузки/file.wav")

data = np.array(signal, dtype=float)
n_data = (data - 128) / 128
analytic_signal = hilbert(n_data)
amplitude_envelope = np.abs(analytic_signal)


length = fs//2



matrix = []
for i in range (len(signal) // (length)):
    matrix_column = []
    for j in range(i * length, i * length + length):
        matrix_column.append(amplitude_envelope[j])
    matrix.append(matrix_column)


matrix = np.array(matrix)
standard = matrix[730][1210:1291]

for i in range(0, len(matrix)):
    arr_delta = []
    for j in range(0, 1700):
        delta = abs(matrix[i][j:j+len(standard)] - standard)
        arr_delta.append(delta.sum())
    index = arr_delta.index(min(arr_delta))
    matrix[i] = np.roll(matrix[i], -index+length//2)


telemetry = []
for i in range(16):
    telemetry.append(np.mean(matrix[390+i*8][2642:2714]))

k = np.polyfit([max(telemetry), min(telemetry)], [1, 0], 1)

for i in range(len(matrix)):
    for j in range(length):
        matrix[i][j] = np.polyval(k, matrix[i][j])

        if matrix[i][j] >= 1:
            matrix[i][j] = 1

        if matrix[i][j] <= 0:
            matrix[i][j] = 0

matrix = 255 * matrix

# plt.figure(figsize=(20, 14))
#plt.imshow(matrix[40:770], cmap='gray')
plt.imshow(matrix[:-1], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/askar/PyCharmProjects/Project1/first_image.png', bbox_inches='tight', pad_inches=0, dpi=905)
plt.show()


#покраска ...

palette = cv2.imread("WXtoImg-NO.png")
pic = cv2.imread("first_image.png")
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)

for i in range(len(matrix)):
    for j in range(220, 2620):
        pic[i][j] = palette[int(matrix[i][j])][int(matrix[i][j+length//2])]

cv2.imwrite('painted.png', pic)
print(1)

###

start_time = datetime.datetime(2022, 4, 18, 15, 30, 4)  # start_latlon = start_lat_rad, start_lon_rad
end_time = datetime.datetime(2022, 4, 18, 15, 37, 3)   # end_latlon = end_lat_rad, end_lon_rad

orb = Orbital("NOAA-19")
start_lat, start_lon, start_alt = orb.get_lonlatalt(start_time)
end_lat, end_lon, end_alt = orb.get_lonlatalt(end_time)

start_latlon = radians(start_lat), radians(start_lon)
end_latlon = radians(end_lat), radians(end_lon)


###


def draw(img):
    yaw = 0.
    vscale = 1
    hscale = 1

    # Compute the great-circle distance between two points
    # The units of all input and output parameters are radians
    def distance(lat1, lon1, lat2, lon2):
        # https://en.wikipedia.org/w/index.php?title=Great-circle_distance&oldid=749078136#Computational_formulas

        delta_lon = lon2 - lon1

        cos_central_angle = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon)

        if cos_central_angle < -1:
            cos_central_angle = -1

        if cos_central_angle > 1:
            cos_central_angle = 1

        return np.arccos(cos_central_angle)

    height = len(img)

    y_res = distance(*start_latlon, *end_latlon) / height / vscale
    x_res = 0.0005 / hscale

    # Compute azimuth of line between two points
    # The angle between the line segment defined by the points (`lat1`,`lon1`) and (`lat2`,`lon2`) and the North
    # The units of all input and output parameters are radians
    def azimuth(lat1, lon1, lat2, lon2):
        # https://en.wikipedia.org/w/index.php?title=Azimuth&oldid=750059816#Calculating_azimuth

        delta_lon = lon2 - lon1

        return np.arctan2(np.sin(delta_lon), np.cos(lat1) * np.tan(lat2) - np.sin(lat1) * np.cos(delta_lon))

    ref_az = azimuth(*start_latlon, *end_latlon)

    def latlon_to_rel_px(latlon):
        az = azimuth(*start_latlon, *latlon)
        B = az - ref_az

        c = distance(*latlon, *start_latlon)

        if c < -np.pi / 3:
            c = -np.pi / 3

        if c > np.pi / 3:
            c = np.pi / 3

        a = np.arctan(np.cos(B) * np.tan(c))
        b = np.arcsin(np.sin(B) * np.sin(c))

        x = -b / x_res

        # Add the yaw correction value
        # Should be calculating sin(yaw) * x but yaw is always a small value
        y = a / y_res + yaw * x

        return (x, y)

    def draw_line(latlon1, latlon2, r, g, b, a):
        # Convert latlon to (x, y)
        (x1, y1) = latlon_to_rel_px(latlon1)
        (x2, y2) = latlon_to_rel_px(latlon2)

        f = interpolate.interp1d((x1, x2), (y1, y2))
        xar = np.arange(x1, x2)
        dlimg = len(img[0]) / 2080
        bounds_1 = int(dlimg * 456)  # 456
        bounds_2 = int(dlimg * 600)  # 600
        shift_1 = int(dlimg * 539)  # 539
        shift_2 = int(dlimg * 1579)  # 1579
        if (x1 > -bounds_1 and x1 < bounds_1 and y1 > 0. and y1 < height) or (
                x1 > -bounds_2 and x1 < bounds_2 and y1 > 0. and y1 < height):
            for x in xar:
                y = f(x)
                if x > -bounds_1 and x < bounds_1 and y > 0 and y < height:
                    img[int(y), int(x) + shift_1] = [r, g, b]
                    img[int(y), int(x) + shift_2] = [r, g, b]

    def draw_shape(shpfile, r, g, b):
        reader = shapefile.Reader(shpfile)
        for shape in reader.shapes():
            prev_pt = shape.points[0]
            for pt in shape.points:
                draw_line(
                    (pt[1] / 180. * np.pi, pt[0] / 180. * np.pi),
                    (prev_pt[1] / 180. * np.pi, prev_pt[0] / 180. * np.pi),
                    r, g, b, 0
                );
                prev_pt = pt;

    draw_shape("https://github.com/nvkelso/natural-earth-vector/blob/master/10m_cultural/ne_10m_admin_0_countries.shp?raw=true", 255, 255, 0)
    return img


img = draw(pic)
# Displaying the image
cv2.imwrite('/home/askar/PyCharmProjects/Project1/borders.png', img)
#cv2.imshow('window_name', img)
#cv2.waitKey(0)

###
m = cv2.cvtColor(cv2.imread("borders.png"), cv2.COLOR_BGR2RGB)

maps = []
for i in range(len(matrix)):
    maps.append(m[i][220:2620])

plt.imshow(maps)
plt.axis('off')
plt.savefig("borders2.png", bbox_inches='tight', pad_inches=0)


maps = cv2.imread("borders2.png")

ImgX1 = len(maps[1]) / 2
ImgY1 = -1
ImgX2 = len(maps[1]) / 2
ImgY2 = - len(maps)

transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
MapX1, MapY1 = transformer.transform(start_lat, start_lon)
MapX2, MapY2 = transformer.transform(end_lat, end_lon)

f = open("ArcGIS.txt", 'w')
f.write(str(ImgX1))
f.write("\t")
f.write(str(ImgY1))
f.write("\t")
f.write(str(MapX1))
f.write("\t")
f.write(str(MapY1))
f.write('\n')
f.write(str(ImgX2))
f.write("\t")
f.write(str(ImgY2))
f.write("\t")
f.write(str(MapX2))
f.write("\t")
f.write(str(MapY2))
f.close()