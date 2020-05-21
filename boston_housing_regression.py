from sklearn.datasets import load_boston
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import numpy as np

"""
Synopsis:

    1.) Load Boston house price data and rescale to approximately similar
        data ranges using yeo-johnson transform (semi-arbitrarily selected;
        there were other options but I didn't test them).

    2.) Test drive univariate feature ranking algorithms.  These attempt to
        rank features based on how well each one by itself is capable of
        potentially predicting the target variable.

    3.) For benefit of OLS algorithm (which can easily overfit) attempt to
        pick optimal combination of features at each level of model
        complexity, ranging from 1 feature up to all 13.

    4.) Attempt to observe bias-variance tradeoff in OLS model, as number
        of features (and model complexity) increases.  Unfortunately I
        didn't observe this (mean squared error scoring only improves with
        increasing features; does not appear to inflect and begin rising
        again), however I don't think I made a mistake, because intrinsicly
        regularizing models such as ridge or random forest also seem to
        get roughly comparable fit results.

    5.) Show cross-validation based mean squared prediction error for four
        regression models.  Accept all model hyperparameter defaults, don't
        attempt to explore potential hyperparameter optimizations.

    6.) Show scatter between observation and prediction of each data point
        in the test set, using model fit results from training data set.
        (This is essentially a more granular visualization of the same model
        performance data measured in part 5.)
"""

# -------------------------------
# Part 1: Load and Condition Data
# -------------------------------

# Load data, ascertain shape, and assign to standardized variable names
bos = load_boston()
Xraw = bos['data']
nsamp, nfeat = Xraw.shape
y = bos['target']

# Standardize variable ranges to be about the same.  This particular
# transform tends to somewhat preserve outliers, while making them
# less extreme
Xtrans = PowerTransformer(method='yeo-johnson').fit_transform(Xraw)

# Plot all 13 input variables before and after transformation
fgvars = plt.figure(num='Feature Distributions: Raw and Transformed',
                figsize=(8,13.5))
axvars = {}
for ii in range(nfeat):
    # Plot raw distributions
    axvars[ii*2+1] = fgvars.add_subplot(nfeat, 2, ii*2+1)
    axvars[ii*2+1].hist(Xraw[:,ii], bins=50)
    axvars[ii*2+1].set_ylabel(bos['feature_names'][ii])
    # Plot transformed distributions
    axvars[ii*2+2] = fgvars.add_subplot(nfeat, 2, ii*2+2)
    axvars[ii*2+2].hist(Xtrans[:,ii], bins=50)

fgvars.tight_layout()

# -------------------------------------------------------
# Part 2: Rank Features in Univariate Order of Importance
# -------------------------------------------------------

# First do a univariate importance ranking, just to exercise a few of the
# different scoring methods
mutinf = mutual_info_regression(X=Xtrans, y=y)
miidx = np.argsort(mutinf)[::-1]
F, pval = f_regression(X=Xtrans, y=y)
lpv = -np.log10(pval)
fpidx = np.argsort(lpv)[::-1]
# Print ranking.  Note that if two features are themselves highly 
# correlated, they may both have a high ranking, but using the second
# one won't improve fit results much, because it brings only redundant
# information.  So this is not the whole story--we need to choose an
# optimal set (see below).
print('Rank  MutInf_Score  MutInf_Feature  F_logp_Score  F_logp_Feature')
tmpl = ' {:2d}      {:7.5f}         {:<7}       {:7.3f}        {:<7}'
for ii in range(nfeat):
    print(tmpl.format(ii+1, mutinf[miidx[ii]], bos['feature_names'][miidx[ii]], 
                      lpv[fpidx[ii]], bos['feature_names'][fpidx[ii]]))

# -------------------------------------------------------
# Part 3: For Each Number of Features, Choose Optimal Set
# -------------------------------------------------------
kbest = []
# Consider combinations of up to all 13 features
for ii in range(13):
    # Arbitrarily choose mutual information as the scoring metric
    kb = SelectKBest(mutual_info_regression, k=ii+1)
    kb.fit(Xtrans, y)
    kbest.append(kb.get_support())

# ------------------------------------------------------------------------
# Part 4; For OLS, Plot Mean Squared Prediction Error vs. Model Complexity
# ------------------------------------------------------------------------

# Define the cross validation strategy and the fit model
cv = RepeatedKFold()
linreg = LinearRegression()
ftcvscores = []
# Loop over total number of features and attempt to show bias-variance
# tradeoff as OLS model includes more features and increases in complexity
for ii in range(len(kbest)):
    scores = cross_val_score(linreg, Xtrans[:,kbest[ii]], y,
                             scoring='neg_mean_squared_error', cv=cv)
    ftcvscores.append(-scores)

# --------------------------------------------------------------------
# Part 5: Show Test Truth vs. Prediction for Multiple Regression Types
# --------------------------------------------------------------------

# For simplicity, just use default hyperparameters for all models
models = {'OLS': LinearRegression(),
          'Ridge': Ridge(),
          'RandomForest': RandomForestRegressor(),
          'AdaBoost': AdaBoostRegressor()}
mscores = []
names = ['OLS', 'Ridge', 'RandomForest', 'AdaBoost']
# Loop over models and calculate mean squared error scores
for k in names:
    scores = cross_val_score(models[k], Xtrans, y,
                             scoring='neg_mean_squared_error', cv=cv)
    mscores.append(-scores)

# Plot results for both part 5 and part 6 in the same figure
fgmse = plt.figure(num='Prediction Mean Squared Error', figsize=(8,8))
axmse = {}
axmse['OLSfeat'] = fgmse.add_subplot(2,1,1)
axmse['OLSfeat'].boxplot(ftcvscores, labels=[ii+1 for ii in range(len(kbest))],
                         showmeans=True)
axmse['OLSfeat'].set_xlabel('KBest Number of Features')
axmse['OLSfeat'].set_ylabel('Mean Squared Prediction Error')
axmse['models'] = fgmse.add_subplot(2,1,2)
axmse['models'].boxplot(mscores, labels=names, showmeans=True)
axmse['models'].set_xlabel('Regression Model')
axmse['models'].set_ylabel('Mean Squared Prediction Error')
fgmse.tight_layout()

# --------------------------------------------------------------------
# Part 6: Show Typical Scatter Between Test Prediction and Observation
# --------------------------------------------------------------------

# Split into train / test, fit each model using training data, and
# predict values using test data
Xtrn, Xtst, ytrn, ytst = train_test_split(Xtrans, y, test_size=0.5)
ypred = []
for k in names:
    fit = models[k].fit(Xtrn, ytrn)
    ypred.append(fit.predict(Xtst))

# Plot scatter between test set target observation and target prediction 
fgscat = plt.figure(num='Target Predictions vs. Observations', figsize=(8,8))
ax = {}
for ii, nm in enumerate(names, 1):
    ax[ii] = fgscat.add_subplot(2,2,ii)
    ax[ii].scatter(ytst, ypred[ii-1], s=2)
    ax[ii].set_title(nm)
    ax[ii].set_xlabel('Target Observation')
    ax[ii].set_ylabel('Target Prediction')
fgscat.tight_layout()

plt.show()