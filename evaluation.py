import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import collections
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import xgboost as xgb

from FairReg import FairReg
from evaluation_measures import DP_unfairness, prob_unfairness, prob_unfairness_summary, DP_unfairness_summary
from data_prep import get_lawschool_data, get_communities_data, get_frequencies

def get_stats(dataset='lawschool', num=10, T=15000, eps = [0.01, 0.01], beta='auto', L='auto',
              TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2):
   
    if dataset=='lawschool':
        X, S, y = get_lawschool_data()
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        B=4
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
    
    risk_history_all = []
    base_risk_all = []
    max_all, avg_all, sum_all, std_all, DP_all, base_DP_all = {}, {}, {}, {}, {}, {}
    for s in range(K):
        max_all[s], avg_all[s], sum_all[s], std_all[s], DP_all[s], base_DP_all[s] = [], [], [], [], [], []


    
    for i in range(1, num+1):
    
        print (i,'/',num, ': training...')

        X_train, X_, S_train, S_, y_train, y_ = train_test_split(X, S, y, train_size=TRAIN_SIZE, stratify=S)
        X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(X_, S_, y_, test_size = TEST_SIZE/(1-TRAIN_SIZE), stratify=S_)

#         if dataset=='lawschool':
#             reg = xgb.XGBRegressor(max_depth=10, objective='reg:linear', n_estimators=400, reg_lambda=1, gamma=2, verbosity = 0)
#             clf = xgb.XGBClassifier(max_depth=5, objective='binary:logistic', n_estimators=200, gamma=2, verbosity = 0)
#         elif dataset=='communities':

        reg = LinearRegression(fit_intercept=False)
        clf = LogisticRegression()
            
        reg.fit(X_train,y_train)
        clf.fit(X_train,S_train)

        fair_reg = FairReg(reg, clf, B=B, K=K,  p=p, eps=eps, T=T, keep_history = True)
        fair_reg.fit(X_unlab, beta, L)

        base_risk_all.append(mse(y_test, reg.predict(X_test)))
        base_DP = DP_unfairness(reg.predict(X_test), S_test, bins=fair_reg.Q_L)

        print ('training completed')
        print ('retrieving training history...')

        risk_history, prob_unfairness_history, DP_unfairness_history = fair_reg.history(X_test, S_test, y_test)
        risk_history_all.append(risk_history)

        max_prob_unf, avg_prob_unf, sum_prob_unf, std_prob_unf = prob_unfairness_summary(prob_unfairness_history, fair_reg.K)
        DP_unf = DP_unfairness_summary(DP_unfairness_history, fair_reg.K)

        max_arr, avg_arr, sum_arr, std_arr, DP_arr, base_DP_arr = [], [], [], [], [], []
        for s in range(K):
            max_all[s].append(max_prob_unf[s])
            max_arr.append(max_prob_unf[s])

            avg_all[s].append(avg_prob_unf[s])
            avg_arr.append(avg_prob_unf[s])

            sum_all[s].append(sum_prob_unf[s])
            sum_arr.append(sum_prob_unf[s])

            std_all[s].append(std_prob_unf[s])
            std_arr.append(std_prob_unf[s])

            DP_all[s].append(DP_unf[s])
            DP_arr.append(DP_unf[s])
            
            base_DP_all[s].append(base_DP[s])
            base_DP_arr.append(base_DP[s])

        #saving the arrays for reproducing
        np.save('./saved_arrays/'+dataset+'_max_num_'+str(i), max_arr)
        np.save('./saved_arrays/'+dataset+'_avg_num_'+str(i), avg_arr)
        np.save('./saved_arrays/'+dataset+'_sum_num_'+str(i), sum_arr)
        np.save('./saved_arrays/'+dataset+'_std_num_'+str(i), std_arr)
        np.save('./saved_arrays/'+dataset+'_DP_num_'+str(i), DP_arr)
        np.save('./saved_arrays/'+dataset+'_risk_num_'+str(i), risk_history_all)
        np.save('./saved_arrays/'+dataset+'_base_DP_num_'+str(i), base_DP_arr)
        np.save('./saved_arrays/'+dataset+'_base_risk_'+str(i), base_risk_all)

        print ('training history retrieved and saved')
        print ('====================================')
        
        
    return risk_history_all, max_all, avg_all, sum_all, std_all, DP_all, base_risk_all, base_DP_all
    
    
    
    
def get_risk_unf_wrt_eps(dataset, num, T, eps_list, print_details = True, beta='auto', L='auto',
            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2):
    
    if dataset=='lawschool':
        X, S, y = get_lawschool_data()
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        B=4
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
        
    prob_risk_all, mse_risk_all, base_mse_risk_all = [], [], []
    avg_all, DP_all, base_DP_all = {}, {}, {}
    for s in range(K):
        avg_all[s], DP_all[s], base_DP_all[s] = [], [], []
        
    for k, eps in enumerate(eps_list):
        if print_details:
            print (k+1,'/',len(eps_list), ' : collecting statistics for eps='+str(eps))
        
        prob_risk, mse_risk, base_mse_risk = [], [], []
        avg_unf, DP_unf, base_DP_unf = {}, {}, {}
        for s in range(K):
            avg_unf[s], DP_unf[s], base_DP_unf[s] = [], [], []
    
        for i in range(1, num+1):

            X_train, X_, S_train, S_, y_train, y_ = train_test_split(X, S, y, train_size=TRAIN_SIZE, stratify=S)
            X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(X_, S_, y_, test_size = TEST_SIZE/(1-TRAIN_SIZE), stratify=S_)

            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train,y_train)
            
            clf = LogisticRegression()
            clf.fit(X_train,S_train)

            fair_reg = FairReg(reg, clf, B=B, K=K,  p=p, eps=eps, T=T, keep_history = False)
            fair_reg.fit(X_unlab, beta, L)
            
            y_pred_base = reg.predict(X_test)
            y_pred_fair = fair_reg.predict(X_test)
            y_pred_prob_fair = fair_reg.pred_prob
            
            r_X = np.square(y_pred_base[:, np.newaxis] - fair_reg.Q_L)
            prob_risk.append(np.mean(np.sum(r_X*y_pred_prob_fair, axis=1)))
            
            mse_risk.append(mse(y_test, y_pred_fair))
            base_mse_risk.append(mse(y_test, y_pred_base))
            
            prob_unf = prob_unfairness(y_pred_prob_fair, S_test)
            DP = DP_unfairness(y_pred_fair, S_test, bins=fair_reg.Q_L)
            base_DP = DP_unfairness(y_pred_base, S_test, bins=fair_reg.Q_L)
            
            for s in range(K):
                avg_unf[s].append(prob_unf[s].mean()) #the average of prob. unfairness[s] in grid
                DP_unf[s].append(DP[s])
                base_DP_unf[s].append(base_DP[s])
            if print_details:    
                print ('-----   ', i,'/',num,': training completed, statistics collected')
        
        prob_risk_all.append(np.mean(prob_risk))
        mse_risk_all.append(np.mean(mse_risk))
        base_mse_risk_all.append(np.mean(base_mse_risk))        
                
        for s in range(K):
            avg_all[s].append(np.mean(avg_unf[s]))
            DP_all[s].append(np.mean(DP_unf[s]))
            base_DP_all[s].append(np.mean(base_DP_unf[s]))
            
        print ('---------------------------------------------------------')
        
    base_mse_risk = np.mean(base_mse_risk_all)
    for s in range(K):
        base_DP_all[s]=np.mean(base_DP_all[s])
            
    return prob_risk_all, mse_risk_all, base_mse_risk, avg_all, DP_all, base_DP_all
