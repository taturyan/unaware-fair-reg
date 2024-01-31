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

def prob_unfairness(pred_prob, S):
    mean = pred_prob.mean(axis=0)
    Unfairness = {}
    S_val = sorted(S.unique())
    for s in S_val:
        mean_s = pred_prob[S==s].mean(axis=0)
        Unfairness[s]=abs(mean_s-mean)
    return Unfairness

def prob_unfairness_summary(prob_unfairness_history, K):

    max_prob_unf = {}
    avg_prob_unf = {}
    sum_prob_unf = {}
    std_prob_unf = {}

    for s in range(K):
        max_prob_unf[s] = []
        avg_prob_unf[s] = []
        sum_prob_unf[s] = []
        std_prob_unf[s] = []

    for elem in prob_unfairness_history:
        for s in range(K):            
            max_prob_unf[s].append(max(elem[s]))
            avg_prob_unf[s].append(elem[s].mean()) 
            sum_prob_unf[s].append(sum(elem[s]))
            std_prob_unf[s].append(np.std(elem[s]))
            
    return max_prob_unf, avg_prob_unf, sum_prob_unf, std_prob_unf

def DP_unfairness_summary(DP_unfairness_history, K):
    
    DP_unf = {}
    for s in range(K):
        DP_unf[s] = []
    for elem in DP_unfairness_history:
        for s in range(K):
            DP_unf[s].append(elem[s])
            
    return DP_unf

def prob_risk(base_pred, grid, pred_prob):
    
    r_X = np.square(base_pred[:, np.newaxis] - grid)
    risk = np.mean(np.sum(r_X*pred_prob, axis=1))