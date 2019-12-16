import numpy as np
import scipy.io.wavfile as wf
from ica import ICA

rate1, data1 = wf.read('./sounds/mix_1.wav')
rate2, data2 = wf.read('./sounds/mix_2.wav')
rate3, data3 = wf.read('./sounds/mix_3.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

data = [data1.astype(float), data2.astype(float), data3.astype(float)]

y = ICA(data).ica()
y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('music1.wav', rate1, y[0])
wf.write('music2.wav', rate2, y[1])
wf.write('music3.wav', rate3, y[2])
