import numpy as np

def DP_unfairness(y, S, bins='auto'):
    
    hist, bin_edges = np.histogram(y,bins=bins)
    CDF = np.cumsum(hist/sum(hist))
    
    Unfairness = {}
    S_val = sorted(S.unique())
    for s in S_val:
        hist, bin_edges = np.histogram(y[S==s],bins=bins)
        CDF_s = np.cumsum(hist/sum(hist))
        Unfairness[s]=max(abs(CDF_s-CDF))
    return Unfairness



def probabilistic_pred_unfairness(fair_model, X, S):
    y_pred = fair_model.predict(X)
    mean = fair_model.pred_prob.mean(axis=0)
    Unfairness = {}
    S_val = sorted(S.unique())
    for s in S_val:
        mean_s = fair_model.pred_prob[S==s].mean(axis=0)
        Unfairness[s]=abs(mean_s-mean)
    return Unfairness