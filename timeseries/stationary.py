# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:02:34 2018

@author: TH
"""

import pylab as pl
import pandas as  pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

t = np.arange(0, 1000000) * 0.1
x = np.sin(t)
n = wgn(x, 6)
xn = x+n # 增加了6dBz信噪比噪声的信号
pl.subplot(211)
pl.hist(n, bins=10, normed=True)
pl.subplot(212)
pl.psd(n)
pl.show()
pl.acorr(np.ones(20))
pl.show()

plot_acf(np.ones(20))
