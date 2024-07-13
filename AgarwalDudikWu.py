#The functions in this file are copied from "https://github.com/steven7woo/fair_regression_reduction/" for comparison purposes. We do a slight modification in the part of "DP disp" in the function "evaluate_FairModel" to present DP disparity for all sensitivie groups. We also sort the names of sensitive "groups".

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import collections
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
import time

import scipy.optimize as opt
import pickle
import functools
from collections import namedtuple
import scipy
from scipy.stats import norm

import random
#from gurobipy import *

from itertools import repeat
import itertools
print = functools.partial(print, flush=True)
"""
Augment the dataset according to the loss functions.

Input:
- a regression data set (x, a, y), which may be obtained using the data_parser
- loss function
- Theta, a set of thresholds in between 0 and 1

Output:
a weighted classification dataset (X, A, Y, W)
"""

_LOGISTIC_C = 5



def augment_data_ab(X, A, Y, Theta):
    """
    Takes input data and augment it with an additional feature of
    theta; Return: X tensor_product Theta
    For absolute loss, we don't do any reweighting.  
    TODO: might add the alpha/2 to match with the write-up
    """
    n = np.shape(X)[0]
    num_theta = len(Theta)
    X_aug = pd.concat(repeat(X, num_theta))
    A_aug = pd.concat(repeat(A, num_theta))
    Y_values = pd.concat(repeat(Y, num_theta))
    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

    Y_aug = Y_values >= X_aug['theta']
    Y_aug = Y_aug.map({True: 1, False: 0})
    X_aug.index = range(n * num_theta)
    Y_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    W_aug = pd.Series(1, Y_aug.index)
    return X_aug, A_aug, Y_aug, W_aug


def augment_data_sq(x, a, y, Theta):
    """
    Augment the dataset so that the x carries an additional feature of theta
    Then also attach appropriate weights to each data point.

    Theta: Assume uniform grid Theta
    """
    n = np.shape(x)[0]  # number of original data points
    num_theta = len(Theta)
    width = Theta[1] - Theta[0]
    X_aug = pd.concat(repeat(x, num_theta))
    A_aug = pd.concat(repeat(a, num_theta))
    Y_values = pd.concat(repeat(y, num_theta))

    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)
    X_aug.index = range(n * num_theta)
    # Y_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    Y_values.index = range(n * num_theta)

    # two helper functions
    sq_loss = lambda a, b: (a - b)**2  # square loss function
    weight_assign = lambda theta, y: (sq_loss(theta + width/2, y) - sq_loss(theta - width/2, y))
    W = weight_assign(X_aug['theta'], Y_values)
    Y_aug = 1*(W < 0)
    W = abs(W)
    # Compute the weights
    return X_aug, A_aug, Y_aug, W


def augment_data_logistic(x, a, y, Theta):
    """
    Augment the dataset so that the x carries an additional feature of theta
    Then also attach appropriate weights to each data point, so that optimize
    for logisitc loss
    
    Theta: Assume uniform grid Theta
    y: assume the labels are {0, 1}
    """
    n = np.shape(x)[0]  # number of original data points
    num_theta = len(Theta)
    width = Theta[1] - Theta[0]
    X_aug = pd.concat(repeat(x, num_theta))
    A_aug = pd.concat(repeat(a, num_theta))
    Y_values = pd.concat(repeat(y, num_theta))

    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

    X_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    Y_values.index = range(n * num_theta)

    # two helper functions
    logistic_loss = lambda y_hat, y: np.log(1 + np.exp(-(_LOGISTIC_C)*(2 * y - 1) * (2 * y_hat - 1))) / (np.log(1 + np.exp(_LOGISTIC_C)))  # re-scaled logistic loss
    #logistic_loss = lambda y_hat, y: np.log(1 + np.exp(-(_LOGISTIC_C)*(2 * y - 1) * (2 * y_hat - 1)))  # re-scaled logistic loss
    weight_assign = lambda theta, y: (logistic_loss(theta + width/2,
                                                    y) - logistic_loss(theta - width/2, y))
    W = weight_assign(X_aug['theta'], Y_values)
    Y_aug = 1*(W < 0)
    W = abs(W)
    # Compute the weights
    return X_aug, A_aug, Y_aug, W



"""
Two types of solvers/optimizers:

1. The first type take in an augmented data set returned by
data_augment, and try to minimize classification error over the
following hypothesis class: { h(X) = 1[ f(x) >= x['theta']] : f in F}
over some real-valued class F.

Input: augmented data set, (X, Y, W)
Output: a model that can predict label Y

These solvers are used with exp_grad

2. The second type simply solves the regression problem
on a data set (x, a, y)

These solvers serve as our unconstrained benchmark methods.
"""






_LOGISTIC_C = 5  # Constant for rescaled logisitic loss; might have to
                 # change for data_augment
# from sklearn.model_selection import train_test_split

"""
Oracles for fair regression algorithm
"""
class SVM_LP_Learner:
    """
    Gurobi based cost-sensitive classification oracle
    Assume there is a 'theta' field in the X data frame
    Oracle=CS; Class=linear
    """
    def __init__(self, off_set=0, norm_bdd=1):
        self.weights = None
        self.norm_bdd = norm_bdd  # initialize the norm bound to be 2
        self.off_set = off_set
        self.name = 'SVM_LP'

    def fit(self, X, Y, W):
        w = SVM_Gurobi(X, Y, W, self.norm_bdd, self.off_set)
        self.weights = pd.Series(w, index=list(X.drop(['theta'], 1)))

    def predict(self, X):
        y_values = (X.drop(['theta'],
                           axis=1)).dot(np.array(self.weights))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class LeastSquaresLearner:
    """
    Basic Least regression square based oracle
    Oracle=LS; class=linear
    """
    def __init__(self, Theta):
        self.weights = None
        self.Theta = Theta
        self.name = "OLS"

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=None)
        self.weights = pd.Series(self.lsqinfo[0], index=list(matX))

    def predict(self, X):
        y_values = (X.drop(['theta'],
                           axis=1)).dot(np.array(self.weights))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

class LogisticRegressionLearner:
    """
    Basic Logistic regression baed oracle
    Oralce=LR; Class=linear
    """
    def __init__(self, Theta, C=10000, regr=None):
        self.Theta = Theta
        self.name = "LR"

        if regr is None:
            self.regr = LogisticRegression(random_state=0, C=C,
                                           max_iter=1200,
                                           fit_intercept=False,
                                           solver='lbfgs')
        else:
            self.regr = regr


    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY, sample_weight=vecW)
        pred_prob = self.regr.predict_proba(matX)

    def predict(self, X):
        pred_prob = self.regr.predict_proba(X.drop(['theta'], axis=1))
        prob_values = pd.DataFrame(pred_prob)[1]
        y_values = (np.log(1 / prob_values - 1) / (- _LOGISTIC_C) + 1) / 2
        # y_values = pd.DataFrame(pred_prob)[1]
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

class RF_Classifier_Learner:
    """
    Basic RF classifier based CSC
    Oracle=LR; Class=Tree ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "RF Classifier"
        self.clf = RandomForestClassifier(max_depth=4,
                                           random_state=0,
                                           n_estimators=20)

    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.clf.fit(matX, vecY, sample_weight=vecW)

    def predict(self, X):
        pred_prob = self.clf.predict_proba(X.drop(['theta'],
                                                   axis=1))
        y_values = pd.DataFrame(pred_prob)[1]
        pred = 1*(y_values - X['theta'] >= 0)
        return pred

class XGB_Classifier_Learner:
    """
    Basic GB classifier based oracle
    Oracle=LR; Class=Tree ensemble
    """
    def __init__(self, Theta, clf=None):
        self.Theta = Theta
        self.name = "XGB Classifier"
        param = {'max_depth' : 3, 'silent' : 1, 'objective' :
                 'binary:logistic', 'n_estimators' : 150, 'gamma' : 2}
        if clf is None:
            self.clf = xgb.XGBClassifier(**param)
        else:
            self.clf = clf

    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.clf.fit(matX, vecY, sample_weight=vecW)

    def predict(self, X):
        pred_prob = self.clf.predict_proba(X.drop(['theta'],
                                                  axis=1))
        prob_values = pd.DataFrame(pred_prob)[1]
        y_values = (np.log(1 / prob_values - 1) / (- _LOGISTIC_C) + 1) / 2
        pred = 1*(y_values - X['theta'] >= 0)
        return pred

class RF_Regression_Learner:
    """
    Basic random forest based oracle
    Oracle=LS; Class=Tree ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "RF Regression"
        self.regr = RandomForestRegressor(max_depth=4, random_state=0,
                                          n_estimators=200)

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class XGB_Regression_Learner:
    """
    Gradient boosting based oracle
    Oracle=LS; Class=Tree Ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "XGB Regression"
        params = {'max_depth': 4, 'silent': 1, 'objective':
                  'reg:linear', 'n_estimators': 200, 'reg_lambda' : 1,
                  'gamma':1}
        self.regr = xgb.XGBRegressor(**params)

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

# HELPER FUNCTIONS HERE FOR BestH Oracles
def SVM_Gurobi(X, Y, W, norm_bdd, off_set):
    """
    Solving SVM using Gurobi solver
    X: design matrix with the last two columns being 'theta'
    A: protected feature
    impose ell_infty constraint over the coefficients
    """
    d = len(X.columns) - 1  # number of predictive features (excluding theta)
    N = X.shape[0]  # number of augmented examples
    m = Model()
    m.setParam('OutputFlag', 0)    
    Y_aug = Y.map({1: 1, 0: -1})
    # Add a coefficient variable per feature
    w = {}
    for j in range(d):
        w[j] = m.addVar(lb=-norm_bdd, ub=norm_bdd,
                        vtype=GRB.CONTINUOUS, name="w%d" % j)
    w = pd.Series(w)

    # Add a threshold value per augmented example
    t = {}  # threshold values
    for i in range(N):
        t[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t%d" % i)
    t = pd.Series(t)
    m.update()
    for i in range(N):
        xi = np.array(X.drop(['theta'], 1).iloc[i])
        yi = Y_aug.iloc[i]
        theta_i = X['theta'][i]
        # Hinge Loss Constraint
        m.addConstr(t[i] >=  off_set - (w.dot(xi) - theta_i) * yi)
    m.setObjective(quicksum(t[i] * W.iloc[i] for i in range(N)))
    m.optimize()
    weights = np.array([w[i].X for i in range(d)])
    return np.array(weights)


def approximate_data(X, Y, W, Theta):
    """
    Given the augmented data (X, Y, W), recover for each example the
    prediction in Theta + alpha/2 that minimizes the cost;
    Thus we reduce the size back to the same orginal size
    """
    n = int(len(X) / len(Theta))  # size of the dataset
    alpha = (Theta[1] - Theta[0])/2
    x = X.iloc[:n, :].drop(['theta'], 1)
    pred_vec = Theta + alpha  # the vector of possible preds

    minimizer = {}

    pred_vec = {}  # mapping theta to pred vector
    for pred in (Theta + alpha):
        pred_vec[pred] = (1 * (pred >= pd.Series(Theta)))

    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]  # the set of rows for i-th example
        W_i = W.iloc[index_set]
        Y_i = Y.iloc[index_set]
        Y_i.index = range(len(Y_i))
        W_i.index = range(len(Y_i))
        cost_i = {}
        for pred in (Theta + alpha):
            cost_i[pred] = abs(Y_i - pred_vec[pred]).dot(W_i)
        minimizer[i] = min(cost_i, key=cost_i.get)
    return x, pd.Series(minimizer)


def approx_data_logistic(X, Y, W, Theta):
    """
    Given the augmented data (X, Y, W), recover for each example the
    prediction in Theta + alpha/2 that minimizes the cost;
    Then create a pair of weighted example so that the prob pred
    will minimize the log loss.
    """
    n = int(len(X) / len(Theta))  # size of the dataset
    alpha = (Theta[1] - Theta[0])/2
    x = X.iloc[:n, :].drop(['theta'], 1)

    pred_vec = {}  # mapping theta to pred vector
    Theta_mid = [0] + list(Theta + alpha) + [1]
    Theta_mid = list(filter(lambda x: x >= 0, Theta_mid))
    Theta_mid = list(filter(lambda x: x <= 1, Theta_mid))

    for pred in Theta_mid:
        pred_vec[pred] = (1 * (pred >= pd.Series(Theta)))

    minimizer = {}
    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]  # the set of rows for i-th example
        W_i = W.iloc[index_set]
        Y_i = Y.iloc[index_set]
        Y_i.index = range(len(Y_i))
        W_i.index = range(len(Y_i))
        cost_i = {}
        for pred in Theta_mid:  # enumerate different possible
                                      # predictions
            cost_i[pred] = abs(Y_i - pred_vec[pred]).dot(W_i)
        minimizer[i] = min(cost_i, key=cost_i.get)

    matX = pd.concat([x]*2, ignore_index=True)
    y_1 = pd.Series(1, np.arange(len(x)))
    y_0 = pd.Series(0, np.arange(len(x)))
    vecY = pd.concat([y_1, y_0], ignore_index=True)
    w_1 = pd.Series(minimizer)
    w_0 = 1 - pd.Series(minimizer)
    vecW = pd.concat([w_1, w_0], ignore_index=True)
    return matX, vecY, vecW


"""
SECOND CLASS OF BENCHMARK SOLVERS
"""

class OLS_Base_Learner:
    """
    Basic OLS solver
    """
    def __init__(self):
        self.regr = linear_model.LinearRegression(fit_intercept=False)
        self.name = "OLS"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        pred = self.regr.predict(x)
        return pred


class SEO_Learner:
    """
    SEO learner by JFS
    """
    def __init__(self):
        self.weights_SEO = None
        self.name = "SEO"

    def fit(self, x, y, sens_attr):
        """
        assume sens_attr is contained in x
        """
        lsqinfo_SEO = np.linalg.lstsq(x, y, rcond=None)
        weights_SEO = pd.Series(lsqinfo_SEO[0], index=list(x))
        self.weights_SEO = weights_SEO.drop(sens_attr)

    def predict(self, x, sens_attr):
        x_res = x.drop(sens_attr, 1)
        pred = x_res.dot(self.weights_SEO)
        return pred


class Logistic_Base_Learner:
    """
    Simple logisitic regression
    """
    def __init__(self, C=10000):
        # use liblinear smaller datasets
        self.regr = LogisticRegression(random_state=0, C=C,
                                       max_iter=1200,
                                       fit_intercept=False,
                                       solver='lbfgs')
        self.name = "LR"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # probabilistic predictions
        pred = self.regr.predict_proba(x)
        return pred

class RF_Base_Regressor: 
    """
    Standard Random Forest Regressor
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=20):
        # initialize a rf learner
        self.regr = RandomForestRegressor(max_depth=max_depth,
                                          random_state=0,
                                          n_estimators=n_estimators)
        self.name = "RF Regressor"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # predictions
        pred = self.regr.predict(x)
        return pred

class RF_Base_Classifier: 
    """
    Standard Random Forest Classifier
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=20):
        # initialize a rf learner
        self.regr = RandomForestClassifier(max_depth=max_depth,
                                           random_state=0,
                                           n_estimators=n_estimators)
        self.name = "RF Classifier"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # predictions
        pred = self.regr.predict_proba(x)
        return pred

class XGB_Base_Classifier:
    """
    Extreme gradient boosting classifier
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=150,
                 gamma=2):
        self.clf = xgb.XGBClassifier(max_depth=max_depth,
                                     silent=1,
                                     objective='binary:logistic',
                                     n_estimators=n_estimators,
                                     gamma=gamma)
        self.name = "XGB Classifier"

    def fit(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        pred = self.clf.predict_proba(x)
        return pred


class XGB_Base_Regressor:
    """
    Extreme gradient boosting regressor
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=4, n_estimators=200):
        param = {'max_depth': max_depth, 'silent': 1, 'objective':
                 'reg:linear', 'n_estimators': n_estimators, 'reg_lambda' : 1, 'gamma':1}
        self.regr = xgb.XGBRegressor(**param)
        self.name = "XGB Regressor"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        pred = self.regr.predict(x)
        return pred


def runtime_test():
    """
    Testing the runtime for different oracles
    Taking 1000 examples from the law school dataset.
    """
    x, a, y = parser.clean_lawschool_full()
    x, a, y = parser.subsample(x, a, y, 1000)
    Theta = np.linspace(0, 1.0, 21)
    X, A, Y, W = augment.augment_data_sq(x, a, y, Theta)
    alpha = (Theta[1] - Theta[0])/2

    start = time.time()
    learner1 = SVM_LP_Learner(off_set=alpha, norm_bdd=1)
    learner1.fit(X, Y, W)
    end = time.time()
    print("SVM", end - start)

    start = time.time()
    learner2 = LeastSquaresLearner(Theta)
    learner2.fit(X, Y, W)
    end = time.time()
    print("OLS", end - start)

    start = time.time()
    learner3 = LogisticRegressionLearner(Theta)
    learner3.fit(X, Y, W)
    end = time.time()
    print("Logistic", end - start)

    start = time.time()
    learner4 = XGB_Regression_Learner(Theta)
    learner4.fit(X, Y, W)
    end = time.time()
    print("XGB least square", end - start)

    start = time.time()
    learner5 = XGB_Classifier_Learner(Theta)
    learner5.fit(X, Y, W)
    end = time.time()
    print("XGB logistic", end - start)
    
    

class Moment:
    """Generic moment"""
    
    def __init__(self, dataX, dataA, dataY):
        self.X = dataX
        self.tags = pd.DataFrame({"attr": dataA, "label": dataY})
        self.n = dataX.shape[0]
        self._gamma_descr = None


class MisclassError(Moment):
    """Misclassification error"""
    short_name = "Err"

    def __init__(self, dataX, dataA, dataY, dataW=None):
        super().__init__(dataX, dataA, dataY)
        if dataW is None:
            self.tags["weight"] = 1
        else:
            self.tags["weight"] = dataW
        self.index = ["all"]

    def gamma(self, predictor):
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags["weight"]*(self.tags["label"]-pred).abs()).mean(),
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def signed_weights(self, lambda_vec=None):
        if lambda_vec is None:
            return self.tags["weight"]*(2*self.tags["label"]-1)
        else:
            return lambda_vec["all"]*self.tags["weight"]*(2*self.tags["label"]-1)


class _CondOpportunity(Moment):
    """Generic fairness metric including DP and EO"""

    def __init__(self, dataX, dataA, dataY, dataGrp):
        super().__init__(dataX, dataA, dataY)
        self.tags["grp"] = dataGrp
        self.prob_grp = self.tags.groupby("grp").size()/self.n
        self.prob_attr_grp = self.tags.groupby(["grp", "attr"]).size()/self.n
        signed = pd.concat([self.prob_attr_grp, self.prob_attr_grp],
                           keys=["+", "-"],
                           names=["sign", "grp", "attr"])
        
        self.index = signed.index
        
    def gamma(self, predictor):
        pred = predictor(self.X)
        self.tags["pred"] = pred
        expect_grp = self.tags.groupby("grp").mean()
        expect_attr_grp = self.tags.groupby(["grp", "attr"]).mean()
        expect_attr_grp["diff"] = expect_attr_grp["pred"] - expect_grp["pred"]
        g_unsigned = expect_attr_grp["diff"]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+","-"],
                             names=["sign", "grp", "attr"])
        self._gamma_descr = str(expect_attr_grp[["pred", "diff"]])
        return g_signed

    def signed_weights(self, lambda_vec):
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        adjust = lambda_signed.sum(level="grp")/self.prob_grp \
                 - lambda_signed/self.prob_attr_grp
        signed_weights = self.tags.apply(
            lambda row: adjust[row["grp"], row["attr"]], axis=1
        )
        return signed_weights
    
    
class DP(_CondOpportunity):
    """Demographic parity"""
    short_name = "DP"

    def __init__(self, dataX, dataA, dataY):
        super().__init__(dataX, dataA, dataY,
                         dataY.apply(lambda y : "all"))

class EO(_CondOpportunity):
    """Equalized odds"""
    short_name = "EO"
    def __init__(self, dataX, dataA, dataY):
        super().__init__(dataX, dataA, dataY,
                         dataY.apply(lambda y : "label="+str(y)))


class DP_theta(_CondOpportunity):
    """DP for regression"""
    short_name = "DP-reg"
    def __init__(self, dataX, dataA, dataY):
        super().__init__(dataX, dataA, dataY,
                         dataX["theta"])
        
        
        
        
"""
This module implements the Lagrangian reduction of fair binary
classification to standard binary classification.

FUNCTIONS
expgrad -- optimize accuracy subject to fairness constraints
"""


__all__ = ["expgrad"]
__version__ = "0.1"
__author__ = "Miroslav Dudik"


print = functools.partial(print, flush=True)

_PRECISION = 1e-12


class _GapResult:
    # The result of a duality gap computation
    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L-self.L_low, self.L_high-self.L)


class _Lagrangian:
    # Operations related to the Lagrangian
    def __init__(self, dataX, dataA, dataY, learner, dataW, cons_class, eps, B,
                 opt_lambda=True, debug=False, init_cache=[]):
        self.X = dataX
        self.obj = MisclassError(dataX, dataA, dataY, dataW)
        self.cons = cons_class(dataX, dataA, dataY)
        self.pickled_learner = pickle.dumps(learner)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.debug = debug
        self.hs = pd.Series()
        self.classifiers = pd.Series()
        self.errors = pd.Series()
        self.gammas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        for classifier in init_cache:
            self.add_classifier(classifier)
        
    def eval_from_error_gamma(self, error, gamma, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        if self.opt_lambda:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_signed.abs())
        else:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_vec)
        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B*(max_gamma-self.eps)
        return L, L_high
    
    def eval(self, h, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        #   gamma -- vector of constraint violations
        #   error -- the empirical error
        
        if callable(h):
            error = self.obj.gamma(h)[0]
            gamma = self.cons.gamma(h)
        else:
            error = self.errors[h.index].dot(h)
            gamma = self.gammas[h.index].dot(h)
        L, L_high = self.eval_from_error_gamma(error, gamma, lambda_vec)
        return L, L_high, gamma, error

    def eval_gap(self, h, lambda_hat, nu):
        # Return the duality gap object for the given h and lambda_hat
        
        L, L_high, gamma, error \
            = self.eval(h, lambda_hat)
        res = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul*lambda_hat)
            if self.debug:
                print("%smul=%.0f" % (" "*9, mul))
            L_low_mul, tmp, tmp, tmp \
                = self.eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if (L_low_mul < res.L_low):
                res.L_low = L_low_mul
            if res.gap() > nu+_PRECISION:
                break
        return res
    
    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_cons = len(self.cons.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_res
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (self.gammas-self.eps, -np.ones((n_cons, 1))), axis=1)
        b_ub = np.zeros(n_cons)
        A_eq = np.concatenate(
            (np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        h = pd.Series(res.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate(
            (-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [
            (None, None) if i==n_cons else (0, None) for i in range(n_cons+1)]
        res_dual = opt.linprog(dual_c, A_ub=dual_A_ub, b_ub=dual_b_ub,
                               bounds=dual_bounds)
        lambda_vec = pd.Series(res_dual.x[:-1], self.cons.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_res = (h, lambda_vec,
                                 self.eval_gap(h, lambda_vec, nu))
        return self.last_linprog_res

    def best_h(self, lambda_vec):
        # Return the classifier that solves the best-response problem
        # for the vector of Lagrange multipliers lambda_vec.
    
        signed_weights = self.obj.signed_weights() \
                         + self.cons.signed_weights(lambda_vec)
        redY = 1*(signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n*redW/redW.sum()

        if self.debug:
            print("%sclassifier start" % ("_"*9,))
        classifier = pickle.loads(self.pickled_learner)
        classifier.fit(self.X, redY, redW)
        self.n_oracle_calls += 1
        if self.debug:
            print("%sclassifier end" % ("_"*9,))
        
        h = lambda X: classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.cons.gamma(h)
        h_val = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            vals =  self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = vals.idxmin()
            best_val = vals[best_idx]
        else:
            best_idx = -1
            best_val = np.PINF

        if h_val < best_val-_PRECISION:
            if self.debug:
                print("%sbest_h: val improvement %f" % ("_"*9, best_val-h_val))
                print("%snclassifiers: %d" % (" "*9, len(self.hs)))
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            best_idx = h_idx

        return self.hs[best_idx], best_idx

    def add_classifier(self, classifier):
        h = lambda X: classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.cons.gamma(h)
        h_idx = len(self.hs)
        self.hs.at[h_idx] = h
        self.classifiers.at[h_idx] = classifier
        self.errors.at[h_idx] = h_error
        self.gammas[h_idx] = h_gamma


def _mean_pred(dataX, hs, weights):
    # Return a weighted average of predictions produced by classifiers in hs
    
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](dataX)
    return pred[weights.index].dot(weights)


### Explicit optimization parameters of expgrad

# A multiplier controlling the automatic setting of nu.
_ACCURACY_MUL = 0.5

# Parameters controlling adaptive shrinking of the learning rate.
_REGR_CHECK_START_T = 5
_REGR_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8

# The smallest number of iterations after which expgrad terminates.
_MIN_T = 0

# If _RUN_LP_STEP is set to True, then each step of exponentiated gradient is
# followed by the saddle point optimization over the convex hull of
# classifiers returned so far.
_RUN_LP_STEP = True


def expgrad(dataX, dataA, dataY, learner, dataW=None, cons_class=DP, eps=0.01,
            T=50, nu=None, eta_mul=2.0, debug=False, init_cache=[]):
    """
    Return a fair classifier under specified fairness constraints
    via exponentiated-gradient reduction.
    
    Required input arguments:
      dataX -- a DataFrame containing covariates
      dataA -- a Series containing the protected attribute
      dataY -- a Series containing labels in {0,1}
      learner -- a learner implementing methods fit(X,Y,W) and predict(X),
                 where X is the DataFrame of covariates, and Y and W
                 are the Series containing the labels and weights,
                 respectively; labels Y and predictions returned by
                 predict(X) are in {0,1}

    Optional keyword arguments:
      cons_class -- the fairness measure (default moments.DP)
      eps -- allowed fairness constraint violation (default 0.01)
      T -- max number of iterations (default 50)
      nu -- convergence threshold for the duality gap (default None,
            corresponding to a conservative automatic setting based on the
            statistical uncertainty in measuring classification error)
      eta_mul -- initial setting of the learning rate (default 2.0)
      debug -- if True, then debugging output is produced (default False)

    Returned named tuple with fields:
      best_classifier -- a function that maps a DataFrame X containing
                         covariates to a Series containing the corresponding
                         probabilistic decisions in [0,1]
      best_gap -- the quality of best_classifier; if the algorithm has
                  converged then best_gap<= nu; the solution best_classifier
                  is guaranteed to have the classification error within
                  2*best_gap of the best error under constraint eps; the
                  constraint violation is at most 2*(eps+best_gap)
      last_t -- the last executed iteration; always last_t < T
      best_t -- the iteration in which best_classifier was obtained
      n_oracle_calls -- how many times the learner was called
      n_classifiers -- how many distinct classifiers have been generated
    """

    ExpgradResult = namedtuple("ExgradResult",
                               "best_classifier best_gap last_t best_t"
                               " n_oracle_calls n_classifiers"
                               " hs classifiers weights")

    n = dataX.shape[0]
    assert dataA.shape[0]==n & dataY.shape[0]==n, \
        "the number of rows in all data fields must match"

    if dataW is None:
        dataW = pd.Series(1, dataY.index)
    else:
        dataW = n*dataW / dataW.sum()

    if debug:
        print("...EG STARTING")

    B = 1/eps
    lagr = _Lagrangian(dataX, dataA, dataY, learner, dataW, cons_class, eps, B,
                       debug=debug, init_cache=init_cache)

    theta  = pd.Series(0, lagr.cons.index)
    Qsum = pd.Series()
    lambdas  = pd.DataFrame()
    gaps_EG = []
    gaps = []
    Qs = []

    last_regr_checked = _REGR_CHECK_START_T
    last_gap = np.PINF
    for t in range(0, T):
        if debug:
            print("...iter=%03d" % t)

        lambda_vec = B*np.exp(theta) / (1+np.exp(theta).sum())
        lambdas[t] = lambda_vec
        lambda_EG = lambdas.mean(axis=1)


        if t == 0:
            h, h_idx = lagr.best_h(0*lambda_vec)
        h, h_idx = lagr.best_h(lambda_vec)

        pred_h = h(dataX)

        if t == 0:
            if nu is None:
                nu = _ACCURACY_MUL * (dataW*(pred_h-dataY).abs()).std() / np.sqrt(n)
            eta_min = nu / (2*B)
            eta = eta_mul / B
            if debug:
                print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                      % (eps, B, nu, T, eta_min))
                #print(lagr.cons.index)

        # if not Qsum.index.contains(h_idx):  # OLD CODE
        if h_idx not in Qsum.index:
            Qsum.at[h_idx] = 0.0
        Qsum[h_idx] += 1.0
        gamma = lagr.gammas[h_idx]

        Q_EG = Qsum / Qsum.sum()
        res_EG = lagr.eval_gap(Q_EG, lambda_EG, nu)
        gap_EG = res_EG.gap()
        gaps_EG.append(gap_EG)
        
        if (t == 0) or not _RUN_LP_STEP:
            gap_LP = np.PINF
        else:
            Q_LP, lambda_LP, res_LP = lagr.solve_linprog(nu)
            gap_LP = res_LP.gap()
            
        if gap_EG < gap_LP:
            Qs.append(Q_EG)
            gaps.append(gap_EG)
        else:
            Qs.append(Q_LP)
            gaps.append(gap_LP)

        if debug:
            print("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                  ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f"
                  % (" "*9, eta, res_EG.L_low, res_EG.L, res_EG.L_high,
                     gap_EG, res_EG.gamma.max(), res_EG.error, gap_LP))

        if (gaps[t] < nu) and (t >= _MIN_T):
            break

        if t >= last_regr_checked*_REGR_CHECK_INCREASE_T:
            best_gap = min(gaps_EG)

            if best_gap > last_gap*_SHRINK_REGRET:
                eta *= _SHRINK_ETA
            last_regr_checked = t
            last_gap = best_gap
            
        theta += eta*(gamma-eps)
        
    last_t = len(Qs)-1
    gaps_series = pd.Series(gaps)
    gaps_best = gaps_series[gaps_series<=gaps_series.min()+_PRECISION]
    best_t = gaps_best.index[-1]
    best_Q = Qs[best_t]
    hs = lagr.hs
    best_classifier = lambda X : _mean_pred(X, hs, best_Q)
    best_gap = gaps[best_t]

    res = ExpgradResult(best_classifier=best_classifier,
                        hs=lagr.hs,
                        classifiers=lagr.classifiers,
                        weights=best_Q,
                        best_gap=best_gap,
                        last_t=last_t,
                        best_t=best_t,
                        n_oracle_calls=lagr.n_oracle_calls,
                        n_classifiers=len(lagr.hs))

    if debug:
        print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
              % (eps, B, nu, T, eta_min))
        print("...last_t=%d, best_t=%d, best_gap=%.6f"
              ", n_oracle_calls=%d, n_hs=%d"
              % (res.last_t, res.best_t, res.best_gap,
                 res.n_oracle_calls, res.n_classifiers))
        tmp, tmp, best_gamma, best_error = lagr.eval(best_classifier, 0*lambda_vec)
        print("...disp=%.6f, err=%.6f"
              % (best_gamma.max(), best_error))

    return res




"""
Run the exponentiated gradient method for training a fair regression
model.

Input:
- (x, a, y): training set
- eps: target training tolerance
- Theta: the set of Threshold
- learner: the regression/classification oracle 
- constraint: for now only handles demographic parity (statistical parity)
- loss: the loss function

Output:
- a predictive model (a distribution over hypotheses)
- auxiliary model info

"""
def train_FairRegression(x, a, y, eps, Theta, learner,
                                constraint="DP", loss="square", init_cache=[]):
    """
    Run fair algorithm on the training set and then record
    the metrics on the training set.

    x, a, y: the training set input for the fair algorithm
    eps: the desired level of fairness violation
    Theta: the set of thresholds (z's in the paper)
    """
    alpha = (Theta[1] - Theta[0])/2

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment_data_sq(x, a, y, Theta)
    elif loss == "absolute":  # absolute loss reweighting (uniform)
        X, A, Y, W = augment_data_ab(x, a, y, Theta)
    elif loss == "logistic":  # logisitic reweighting
        X, A, Y, W = augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))

    if constraint == "DP":  # DP constraint
        result = expgrad(X, A, Y, learner, dataW=W,
                             cons_class=DP_theta, eps=eps,
                             debug=False, init_cache=init_cache)
    else:  # exception
        raise Exception('Constraint not supported: ', str(constraint))


    #print('epsilon value: ', eps, ': number of oracle calls', result.n_oracle_calls)

    model_info = {}  # dictionary for saving data
    model_info['loss_function'] = loss
    model_info['constraint'] = constraint
    model_info['exp_grad_result'] = result
    return model_info





"""
Evaluate the fair model on a dataset;
Also evaluate benchmark algorithms: OLS, SEO, Logistic regression

Main function: evaluate_FairModel
Input:
- (x, a, y): evaluation set (can be training/test set)
- loss: loss function name
- result: returned by exp_grad
- Theta: the set of Threshold

Output:
- predictions over the data set
- weighted loss
- distribution over the predictions
- DP Disparity

TODO: decide the support when we compute disparity
"""


_LOGISTIC_C = 5  # Constant for rescaled logisitic loss
_QEO_EVAL = False  # For now not handling the QEO disparity


def evaluate_FairModel(x, a, y, loss, result, Theta):
    """
    Evaluate the performance of the fair model on a dataset

    Input:
    - X, Y: augmented data
    - loss: loss function name
    - result returned by exp_grad
    - Theta: list of thresholds
    - y: original labels
    """

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment_data_sq(x, a, y, Theta)
    elif loss == "absolute":  # absolute loss reweighting (uniform)
        X, A, Y, W = augment_data_ab(x, a, y, Theta)
    elif loss == "logistic":  # logisitic reweighting
        X, A, Y, W = augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))

    hs = result.hs
    weights = result.weights

    # first make sure the lengths of hs and weights are the same;
    off_set = len(hs) - len(weights)
    if (off_set > 0):
        off_set_list = pd.Series(np.zeros(off_set), index=[i +
                                                           len(weights)
                                                           for i in
                                                           range(off_set)])
        result_weights = weights.append(off_set_list)
    else:
        result_weights = weights

    # second filter out hypotheses with zero weights
    hs = hs[result_weights > 0]
    result_weights = result_weights[result_weights > 0]

    num_h = len(hs)
    num_t = len(Theta)
    n = int(len(X) / num_t)

    # predictions
    pred_list = [pd.Series(extract_pred(X, h(X), Theta),
                           index=range(n)) for h in hs]
    total_pred = pd.concat(pred_list, axis=1, keys=range(num_h))
    # predictions across different groups
    pred_group = extract_group_pred(total_pred, a)


    weighted_loss_vec = loss_vec(total_pred, y, result_weights, loss)

    # Fit a normal distribution to the sq_loss vector
#     loss_mean, loss_std = scipy.stats.norm.fit(weighted_loss_vec)
    loss_mean = weighted_loss_vec.mean()

    # DP disp
    PMF_all = weighted_pmf(total_pred, result_weights, Theta)
    PMF_group = [weighted_pmf(pred_group[g], result_weights, Theta) for g in pred_group]
    DP_disp = [pmf2disp(PMF_g, PMF_all) for PMF_g in PMF_group]
    #avg_weighted_unf = [(abs(PMF_g-PMF_all)).mean() for PMF_g in PMF_group]

    # TODO: make sure at least one for each subgroup
    evaluation = {}
    evaluation['pred'] = total_pred
    evaluation['classifier_weights'] = result_weights
    evaluation['weighted_loss'] = loss_mean
    #evaluation['loss_std'] = loss_std / np.sqrt(n)
    evaluation['disp_std'] = KS_confbdd(n, alpha=0.05)
    evaluation['DP_disp'] = DP_disp
    #evaluation['avg_weighted_unf'] = avg_weighted_unf
    evaluation['n_oracle_calls'] = result.n_oracle_calls

    return evaluation


def eval_BenchmarkModel(x, a, y, model, loss):
    """
    Given a dataset (x, a, y) along with predictions,
    loss function name
    evaluate the following:
    - average loss on the dataset
    - DP disp
    """
    pred = model(x)  # apply model to get predictions
    n = len(y)
    if loss == "square":
        err = mean_squared_error(y, pred)  # mean square loss
    elif loss == "absolute":
        err = mean_absolute_error(y, pred)  # mean absolute loss
    elif loss == "logistic":  # assuming probabilistic predictions
        # take the probability of the positive class
        pred = pd.DataFrame(pred).iloc[:, 1]
        err = log_loss(y, pred, eps=1e-15, normalize=True)
    else:
        raise Exception('Loss not supported: ', str(loss))

    disp = pred2_disp(pred, a, y, loss)

    loss_vec = loss_vec2(pred, y, loss)
    loss_mean, loss_std = norm.fit(loss_vec)

    evaluation = {}
    evaluation['pred'] = pred
    evaluation['average_loss'] = err
    evaluation['DP_disp'] = disp['DP']
    evaluation['disp_std'] = KS_confbdd(n, alpha=0.05)
    evaluation['loss_std'] = loss_std / np.sqrt(n)

    return evaluation


def loss_vec(tp, y, result_weights, loss='square'):
    """
    Given a list of predictions and a set of weights, compute
    (weighted average) loss for each point
    """
    num_h = len(result_weights)
    if loss == 'square':
        loss_list = [(tp.iloc[:, i] - y)**2 for i in range(num_h)]
    elif loss == 'absolute':
        loss_list = [abs(tp.iloc[:, i] - y) for i in range(num_h)]
    elif loss == 'logistic':
        logistic_prob_list = [1/(1 + np.exp(- _LOGISTIC_C * (2 * tp[i]
                                                             - 1))) for i in range(num_h)]
        # logistic_prob_list = [tp[i] for i in range(num_h)]
        loss_list = [log_loss_vec(y, prob_pred, eps=1e-15) for
                     prob_pred in logistic_prob_list]
    else:
        raise Exception('Loss not supported: ', str(loss))
    df = pd.concat(loss_list, axis=1)
    weighted_loss_vec = pd.DataFrame(np.dot(df,
                                            pd.DataFrame(result_weights)))
    return weighted_loss_vec.iloc[:, 0]

def loss_vec2(pred, y, loss='square'):
    """
    Given a list of predictions and a set of weights, compute
    (weighted average) loss for each point
    """
    if loss == 'square':
        loss_vec = (pred - y)**2
    elif loss == 'absolute':
        loss_vec = abs(pred - y)
    elif loss == 'logistic':
        loss_vec = log_loss_vec(y, pred)
    else:
        raise Exception('Loss not supported: ', str(loss))
    return loss_vec


def extract_pred(X, pred_aug, Theta):
    """
    Given a list of pred over the augmented dataset, produce
    the real-valued predictions over the original dataset
    """
    width = Theta[1] - Theta[0]
    Theta_mid = Theta + (width / 2)

    num_t = len(Theta)
    n = int(len(X) / num_t)  # TODO: check whether things divide
    pred_list = [pred_aug[((j) * n):((j+1) * n)] for j in range(num_t)]
    total_pred_list = []
    for i in range(n):
        theta_index = max(0, (sum([p_vec.iloc[i] for p_vec in pred_list]) - 1))
        total_pred_list.append(Theta_mid[theta_index])
    return total_pred_list


def extract_group_pred(total_pred, a):
    """
    total_pred: predictions over the data
    a: protected group attributes
    extract the relevant predictions for each protected group
    """
    groups = sorted(list(pd.Series.unique(a)))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[a == g]
    return pred_per_group


def extract_group_quantile_pred(total_pred, a, y, loss):
    """
    total_pred: a list of prediction Series
    a: protected group attributes
    y: the true label, which also gives us the quantile assignment
    """
    if loss == "logistic":
        y_quant = y  # for binary prediction task, just use labels
    else:
        y_quant = augment.quantization(y)

    groups = sorted(list(pd.Series.unique(a)))
    quants = sorted(list(pd.Series.unique(y_quant)))

    pred_group_quantile = {}
    pred_quantile = {}
    for q in quants:
        pred_quantile[q] = total_pred[y_quant == q]
        for g in groups:
            pred_group_quantile[(g, q)] = total_pred[(a == g) & (y_quant == q)]
    return pred_quantile, pred_group_quantile


def weighted_pmf(pred, classifier_weights, Theta):
    """
    Given a list of predictions and a set of weights, compute pmf.
    pl: a list of prediction vectors
    result_weights: a vector of weights over the classifiers
    """
    width = Theta[1] - Theta[0]
    theta_indices = pd.Series(Theta + width/2)
    weights = list(classifier_weights)
    weighted_histograms = [(get_histogram(pred.iloc[:, i],
                                          theta_indices)) * weights[i]
                           for i in range(pred.shape[1])]

    theta_counts = sum(weighted_histograms)
    pmf = theta_counts / sum(theta_counts)
    return pmf


def get_histogram(pred, theta_indices):
    """
    Given a list of discrete predictions and Theta, compute a histogram
    pred: discrete prediction Series vector
    Theta: the discrete range of predictions as a Series vector
    """
    theta_counts = pd.Series(np.zeros(len(theta_indices)))
    for theta in theta_indices:
        theta_counts[theta_indices == theta] = len(pred[pred == theta])
    return theta_counts

def pmf2disp(pmf1, pmf2):
    """
    Take two empirical PMF vectors with the same support and calculate
    the K-S stats
    """
    cdf_1 = pmf1.cumsum()
    cdf_2 = pmf2.cumsum()
    diff = cdf_1 - cdf_2
    diff = abs(diff)
    return max(diff)


def pred2_disp(pred, a, y, loss):
    """
    Input:
    pred: real-valued predictions given by the benchmark method
    a: protected group memberships
    y: labels
    loss: loss function names (for quantization)

    Output: the DP disparity of the predictions

    TODO: use the union of the predictions as the mesh
    """
    Theta = sorted(set(pred))  # find the support among the predictions
    theta_indices = pd.Series(Theta)

    if loss == "logistic":
        y_quant = y  # for binary prediction task, just use labels
    else:
        y_quant = augment.quantization(y)

    groups = sorted(list(pd.Series.unique(a)))
    quants = sorted(list(pd.Series.unique(y_quant)))

    # DP disparity
    histogram_all = get_histogram(pred, theta_indices)
    PMF_all = histogram_all / sum(histogram_all)
    DP_disp_group = {}
    for g in groups:
        histogram_g = get_histogram(pred[a == g], theta_indices)
        PMF_g = histogram_g / sum(histogram_g)
        DP_disp_group[g] = pmf2disp(PMF_all, PMF_g)


    disp = {}
    disp['DP'] = max(DP_disp_group.values())

    return disp


def log_loss_vec(y_true, y_pred, eps=1e-15):
    """
    return the vector of log loss over the examples
    """
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    trans_label = pd.concat([1-y_true, y_true], axis=1)
    loss = -(trans_label * np.log(y_pred)).sum(axis=1)
    return loss


def KS_confbdd(n, alpha=0.05):
    """
    Given sample size calculate the confidence interval width on K-S stats
    n: sample size
    alpha: failure prob
    ref: http://www.math.utah.edu/~davar/ps-pdf-files/Kolmogorov-Smirnov.pdf
    """
    return np.sqrt((1/(2 * n)) * np.log(2/alpha))




