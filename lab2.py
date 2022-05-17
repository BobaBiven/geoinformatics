import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import math

wav = wavfile.read("/home/askar/Загрузки/signal.wav")
signal = wav[1]
fs = wav[0]


data = np.array(signal, dtype=float)
n_data = (data - 128) / 128
analytic_signal = hilbert(n_data)
amplitude_envelope = np.abs(analytic_signal)

print(np.mean(amplitude_envelope))

length = fs//2

matrix = []
for i in range (len(signal) // (length)):
    matrix_column = []
    for j in range(i * (length), i * (length) + (length)):
        matrix_column.append(amplitude_envelope[j])
    matrix.append(matrix_column)


matrix = np.array(matrix)
standard = amplitude_envelope[3*10**6+3000-100+132:3*10**6+3000-100+208]

for i in range(0, len(matrix)):
    arr_delta = []
    for j in range(0, length - len(standard)):
        delta = abs(matrix[i][j:j+len(standard)] - standard)
        arr_delta.append(delta.sum())
    index = arr_delta.index(min(arr_delta))
    matrix[i] = np.roll(matrix[i], -index + length//2)

telemetry = []
for i in range(16):
    telemetry.append(np.mean(matrix[654+i*8][2631:2741]))

#проверям светлый пиксель (должно быть 7)
print(telemetry.index(max(telemetry)))


k = np.polyfit([max(telemetry), min(telemetry)], [1, 0], 1)
print(max(telemetry))

for i in range(len(matrix)):
    for j in range(length):
        matrix[i][j] = np.polyval(k, matrix[i][j])
        if matrix[i][j] < min(telemetry):
             matrix[i][j] = 0
        if matrix[i][j] > max(telemetry):
             matrix[i][j] = 1

matrix = matrix * 255

plt.imshow(matrix, cmap="gray")
plt.savefig("/home/askar/PyCharmProjects/Project1/check.png")
plt.show()


C1 = telemetry[9]
C2 = telemetry[10]
C3 = telemetry[11]
C4 = telemetry[12]
Cb = telemetry[14]
Cs = matrix[707][2904]

T1 = 272.6067 + 0.051111 * C1 + 1.405783 * 10 ** (-6) * C1 ** 2 + 0
T2 = 272.6119 + 0.051090 * C2 + 1.496037 * 10 ** (-6) * C2 ** 2 + 0
T3 = 272.6311 + 0.051033 * C3 + 1.49699 * 10 ** (-6) * C3 ** 2 + 0
T4 = 272.6268 + 0.051058 * C4 + 1.49311 * 10 ** (-6) * C4 ** 2 + 0

T_mean = np.mean([T1, T2, T3, T4])

A = 1.67396
B = 0.997364

T_mean_ = A + B*T_mean

c1 = 1.1910427*10**-5
c2 = 1.4387752
u_e = u_c = 2670

N_bb = (c1 * u_e**3) / (math.exp(c2*u_c/T_mean_) - 1)

C_E = matrix[400][1400]

N_e = N_bb * (Cs - C_E) / (Cs - Cb)
print(N_e)
T_e_ = c2 * u_c / (math.log(1+(c1 * u_c**3 / N_e)))

T_e = (T_e_ - A) / B

print(T_e, "- конечная температура")