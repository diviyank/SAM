
import numpy as np
import pylab as pl
from numpy.random import uniform
from numpy import exp
from scipy.special import expit as sigmoid, logit



# These are the sigmoid parameters we're going to sample from.
n = 10000
X = np.linspace(-5, 5, n)

# number of runs to average over.
R = 10000

# Used for plotting average p(Y=1)
F = np.zeros_like(X)

# Temporary array for saving on memory allocation, cf. method slow-2.
tmp = np.empty(n)

for _ in range(R):
    # Let's use the same random variables for all methods. This allows
    # for a lower variance comparsion and equivalence testing.
    u = uniform(0, 1, size=n)
    z = logit(u)  # used in fast method: precompute expensive stuff.

    # Requires computing sigmoid for each x.

    print("OK1")
    f = X > z

    print(f)

    # print("OK2")
    # sigmoid(X, out=tmp)
    # s2 = tmp > u
    #
    # print("OK3")
    # s3 = 1 / (1 + exp(-X)) > u
    #
    # print("OK4")
    # f = X > z

    # F += f / R
    # assert (s1 == f).all()
    # assert (s2 == f).all()
    # assert (s3 == f).all()

pl.plot(X, F)
pl.plot(X, sigmoid(X), c='r', lw=2)
