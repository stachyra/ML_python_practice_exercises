#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
Synopsis:

    1.) Generate some synthetic 1D truth data, in the form of a sin curve

    2.) Fit the synthetic sin curve data to the "wrong" function type:
        fit to a high order polynomial rather than another sin wave

            a.) Use a ridge regression, selecting the "optimal" fit as the
                one with the smallest mean squared error between predicted
                vs. truth values in the test set

            b.) Use a lasso regression, selecting the "optimal" fit in the
                same manner

    3.) Plot the results

Notes:

    - The Ridge regression seems to be highly sensitive to the choice of
      solver.  I was only able to get reasonable results using singular
      value decomposition (most other methods either return a singular
      matrix error, or else generate curves that don't look remotely right).

    - Despite quite a lot of experimentation and tuning, I can't get the
      Lasso regression to stop issuing convergence failure warnings.
      However, in spite of this, the actual solutions provided by Lasso
      look intuitively reasonable: the fitted curve matches the truth curve
      fairly well.
"""

# -------------------------------
# Part 1: Generate Synthetic Data
# -------------------------------

# Example strategy: we'll generate synthetic data using a sin() curve, but
# then fit the data to a high degree polynomial

# Number of synthetic data points
n = 200
# Degree of fitted polynomial 
d = 10
# Target data noise std dev
tgtsd = 0.2
# Predictor data range
xrg = [0, 3 * np.pi]
# Grid size for printing truth curve
gsz = 101
# Number of cross validation folds to use
k = 10

# Generate unnoised "truth" data on a 1D grid (for plotting purposes only)
xgrid = np.linspace(xrg[0], xrg[1], gsz)
ygrid = np.sin(xgrid)

# Generate predictor data points uniformly random in desired range
x = xrg[0] + xrg[1] * np.random.rand(n)
# Generate polynomial values, up to degree d
X, Xgrid = np.ones((n,d+1)), np.ones((gsz,d+1))
for ii in range(0, (d+1)):
    # Training / test observations
    X[:,ii] = np.power(x, ii)
    # Grid of truth values, for plotting purposes only
    Xgrid[:,ii] = np.power(xgrid, ii)
# Generate synthetic target observations
y = np.sin(x) + tgtsd * np.random.randn(n)

# -----------------------------------------------------
# Part 2: Generate Ridge & Lasso Regression Fit Results
# -----------------------------------------------------

# Regularization parameter
alpha = []
# Model object
clf = {'ridge': None, 'lasso': None}
# Model coefficients
coef = {'ridge': [], 'lasso': []}
# Mean squared error when predicting test data
mse = {'ridge': [], 'lasso': []}
# Overall minimum mse,
minmse = {'ridge': None, 'lasso': None}
# Alpha value at which minimum mse is observed
alphamin = {'ridge': None, 'lasso': None}
# Loop execution flag value
firsttime = {'ridge': True, 'lasso': True}
# Predicted y values at xgrid loci, for lowest value of alpha, optimum value
# of alpha, and highest value of alpha
yfit = {'ridge': {'lo': None, 'opt': None, 'hi': None},
        'lasso': {'lo': None, 'opt': None, 'hi': None}}

# Analyze minimum mean squared error by k-fold cross validation.  Set shuffle
# to True to randomize how the folds are divided on each iteration over
# regularization parameter (alpha) size (this makes the MSE-vs-alpha curve
# look choppier and more noisy).
kf = KFold(n_splits=k, shuffle=True)

# Loop through each regularization parameter, obtain best fit coefficients,
# and save results
for a in np.logspace(-4, 1, 101):
    alpha.append(a)
    # Use singular value decomposition, because the default method seems to
    # result internally somehow in a singular matrix.  Also, don't fit
    # the intercept, as we will do it "effectively" by creating a dummy
    # column on the left side of the matrix populated with ones.
    clf['ridge'] = Ridge(a, fit_intercept=False, solver='svd')
    # Lasso does not seem to allow choice of solver
    clf['lasso'] = Lasso(a, fit_intercept=False, max_iter=10000,
                         warm_start=True, selection='random')
    for k in ['ridge', 'lasso']:
        # At each alpha value, obtain k estimates of test set mean squared error
        msedist = []
        for trnidx, tstidx in kf.split(X, y):
            clf[k].fit(X[trnidx,:], y[trnidx])
            ypredict = clf[k].predict(X[tstidx,:])
            msedist.append(mean_squared_error(y[tstidx], ypredict))
        mse[k].append(np.mean(msedist))
        # Fit coefficients using all data
        clf[k].fit(X,y)
        coef[k].append(clf[k].coef_)
        # Initialize variables
        if firsttime[k] is True:
            firsttime[k] = False
            minmse[k], alphamin[k] = mse[k][-1], alpha[-1]
            # Rmemeber lowest alpha as an extremum case
            yfit[k]['lo'] = clf[k].predict(Xgrid)
            # Initialize optimum result, to be reassigned later when actual
            # minimum mse value is identified
            yfit[k]['opt'] = yfit[k]['lo']
        else:
            # Search for minimum mean squared error in test set and remember it
            if mse[k][-1] < minmse[k]:
                minmse[k], alphamin[k] = mse[k][-1], alpha[-1]
                yfit[k]['opt'] = clf[k].predict(Xgrid)
# Remember high alpha cases as an opposite extremum
for k in ['ridge', 'lasso']:
    yfit[k]['hi'] = clf[k].predict(Xgrid)

# --------------------
# Part 3: Plot Results
# --------------------

ax = {'ridge': {}, 'lasso': {}}
f = plt.figure(figsize=[8, 10])
for k, ii in zip(['ridge', 'lasso'], [0,1]):
    ax[k]['result'] = f.add_subplot(3, 2, 1+ii)
    # Truth curve
    ax[k]['result'].plot(xgrid, ygrid, '-', color=[0.7, 0.7, 0.7],
                         label='Truth')
    # Noisy observations
    ax[k]['result'].plot(x, y, 'o', color=[0, 0, 1], markersize=1.5,
                         label='Observations')
    # Best fit with optimal alpha
    ax[k]['result'].plot(xgrid, yfit[k]['opt'], '-', color=[1, 0, 0],
                         label='Optimal')
    # Best fits with extreme suboptimal alpha
    ax[k]['result'].plot(xgrid, yfit[k]['lo'], '--', color=[0.8, 0.6, 0],
                         label='Low Alpha')
    ax[k]['result'].plot(xgrid, yfit[k]['hi'], '-.', color=[0, 0.7, 0.7],
                         label='High Alpha')
    ax[k]['result'].set_xlabel('Predictor Value')
    if k == 'ridge':
        ax[k]['result'].set_ylabel('Target Value')
    ax[k]['result'].legend(loc='lower left')

    ax[k]['mse'] = f.add_subplot(3, 2, 3+ii)
    ax[k]['mse'].plot(alpha, mse[k], '-')
    ax[k]['mse'].set_xscale('log')
    ax[k]['mse'].axvline(alphamin[k], linestyle=':')
    if k == 'ridge':
        ax[k]['mse'].set_ylabel('K-fold Mean of Mean Sq Test Pred Error')

    ax[k]['coef'] = f.add_subplot(3, 2, 5+ii)
    ax[k]['coef'].plot(alpha, coef[k], '-')
    ax[k]['coef'].set_xscale('log')
    ax[k]['coef'].axvline(alphamin[k], linestyle=':')
    ax[k]['coef'].set_xlabel('Alpha (Regularization Scale Parameter)')
    if k == 'ridge':
        ax[k]['coef'].set_ylabel('Coefficient Values')

f.tight_layout()

plt.show()