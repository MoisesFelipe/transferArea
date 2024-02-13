# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:58:38 2023

@author: 334256
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
# from pylbfgs import owlqn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#%% generate some data with noise

x = np.sort(np.random.uniform(0, 10, 15))
y = 3 + 0.2 * x + 0.1 * np.random.randn(len(x))


# find L1 line fit
l1_fit = lambda x0, x, y: np.sum(np.abs(x0[0] * x + x0[1] - y))
xopt1 = spopt.fmin(func=l1_fit, x0=[1, 1], args=(x, y))+y[0]
# xopt1[0]+=y[0]
# xopt1[1]+=y[-1]
# xopt1  = (xopt1 - min(y)) / (max(y) - min(y))

# find L2 line fit
l2_fit = lambda x0, x, y: np.sum( np.power(x0[0] * x + x0[1] - y, 2) )
xopt2 = spopt.fmin(func=l2_fit, x0=[1, 1], args=(x, y))+y[0]
# xopt2[0]+=y[1]
# xopt2[1]+=y[-1]
# xopt2 = abs((xopt2 - min(y)) / (max(y) - min(y)))

plt.figure()
plt.plot([x[0],x[-1]],xopt1,color='k')
plt.plot([x[0],x[-1]],xopt2,color='b',linestyle='--')
plt.plot(x,y,color='b',linestyle='none',marker='o')


#%% generate some data with noise

x = np.sort(np.random.uniform(0, 10, 15))
y = 3 + 0.2 * x + 0.1 * np.random.randn(len(x))
y2 = y.copy()
y2[3] += 4
y2[13] -= 3


# find L1 line fit
xopt12 = spopt.fmin(func=l1_fit, x0=[1, 1], args=(x, y2))+y[0]

# find L2 line fit
xopt22 = spopt.fmin(func=l2_fit, x0=[1, 1], args=(x, y2))+y[0]


plt.figure()
plt.plot([x[0],x[-1]],xopt12,color='k')
plt.plot([x[0],x[-1]],xopt22,color='b',linestyle='--')
plt.plot(x,y,color='b',linestyle='none',marker='o')

#%% sum of two sinusoids
n = 5000
t = np.linspace(0, 1/8, n)
y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
yt = spfft.dct(y, norm='ortho')


plt.figure()
plt.subplot(2,1,1)
plt.plot(t,y,color='b')
plt.xlabel("time")
plt.subplot(2,1,2)
plt.plot(1/t,yt,color='b')
plt.xlim([0,300])


# extract small sample of signal
m = int(len(y)*0.1) # 10% sample
ri = np.random.choice(n, m, replace=False) # random sample of indices
ri.sort() # sorting not strictly necessary, but convenient for plotting
t2 = t[ri]
y2 = y[ri]


plt.figure()
plt.subplot(2,1,1)
plt.plot(t,y,color='b')
plt.plot(t2,y2,color='r',linestyle='none',marker="o",markersize=4)
plt.xlabel("time")
plt.xlim([0,0.04])


#%% create idct matrix operator
A = spfft.idct(np.identity(n), norm='ortho', axis=0)
A = A[ri]

# do L1 optimization
vx = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == y2]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

#%% reconstruct signal
x = np.array(vx.value)
x = np.squeeze(x)
yhat = spfft.idct(x, norm='ortho', axis=0)
ythat = spfft.dct(yhat, norm='ortho')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,yhat,color='b')
plt.xlabel("time")
plt.subplot(2,1,2)
plt.plot(1/t,ythat,color='b')
plt.xlim([0,300])

