from pywt import wavedec, waverec
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log

res = 10000
x = np.linspace(-10, 10, res)
signal = np.sinc(x)
noise = np.random.normal(scale=.1, size=res)
sequence = signal + noise

fig1, ax1 = plt.subplots()
ax1.plot(x, sequence)
ax1.plot(x, signal)


levels = 3

w = 'db1'
coeffs = wavedec(sequence, w, level=levels)
print(coeffs)
coeffs_filtered = coeffs

fig2, ax2 = plt.subplots(levels+1,1)

for i in range(levels+1):
    data = coeffs[levels-i]
    if i < levels:
        thresh = 0.3
        data = np.array([np.sign(x)*max(0, abs(x)-thresh) for x in coeffs[levels-i]])
    coeffs_filtered[levels-i] = data
    ax2[i].plot(coeffs[levels-i])
    ax2[i].plot(coeffs_filtered[levels-i])

print(coeffs_filtered)
rec_signal = waverec(coeffs_filtered, w)
ax1.plot(x, rec_signal)

plt.show()