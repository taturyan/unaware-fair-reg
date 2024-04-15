import AgarwalDuduikWu as ADW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import collections
import time
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
        B=4
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        B=1
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
        B=4
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        B=1
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



def compare_with_ADW(dataset, num, T, eps_list, print_details = True, beta='auto', L='auto',
            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2):
    
    #geting the data
    if dataset=='lawschool':
        X_all, S_all, y_all = get_lawschool_data(as_df=True)
        sample_size = 2000 #we take only 2000 samples for comparison
        X, _, S, _, y ,_ = train_test_split(X_all, S_all, y_all, test_size=1 - sample_size / (len(X_all)))
        y = y/4 # normalizing data to [0,1]
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data(as_df=True)
        B=1
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
    ####################################
        
    #initializations
    time_hist, time_hist_ADW = [], []
    prob_risk_all, mse_risk_all, loss_ADW_all, base_mse_risk_all = [], [], [], []
    avg_all, sum_all, DP_all, unf_ADW_all, base_DP_all = {}, {}, {}, {}, {}
    for s in range(K):
        avg_all[s], sum_all[s], DP_all[s], unf_ADW_all[s], base_DP_all[s] = [], [], [], [], []
    ####################################
        
    #params for Agarwal et.al    
    Theta = np.linspace(0, 1.0, 41)
    alpha = (Theta[1] - Theta[0])/2
    _SMALL = False  # small scale dataset for speed and testing
    learner=ADW.LeastSquaresLearner(Theta)
    ######################################
        
    for k, eps in enumerate(eps_list):
        if print_details:
            print (k+1,'/',len(eps_list), ' : collecting statistics for eps='+str(eps))
        
        loss_ADW = []
        prob_risk, mse_risk, base_mse_risk = [], [], []
        avg_unf, sum_unf, DP_unf, unf_ADW, base_DP_unf = {}, {}, {}, {}, {}
        for s in range(K):
            avg_unf[s], sum_unf[s], DP_unf[s], unf_ADW[s], base_DP_unf[s] = [], [], [], [], []
     
        for i in range(1, num+1):
            
            
            X_train_df, X_test_df, S_train_df, S_test, y_train_df, y_test = train_test_split(X, S, y, 
                                                                                             test_size=TEST_SIZE, stratify=S,
                                                                                             random_state=i)
            X_train_df.index, S_train_df.index, y_train_df.index = range(len(X_train_df)), range(len(S_train_df)), range(len(y_train_df))
            X_test_df.index, S_test.index, y_test.index = range(len(X_test_df)), range(len(S_test)), range(len(y_test))

            #additionally splitting into train and unlab for our method 
            X_train, X_unlab, S_train, S_unlab, y_train, y_unlab = train_test_split(X_train_df.to_numpy(), S_train_df, y_train_df, 
                                                                        train_size = TRAIN_SIZE/(1-TEST_SIZE), stratify=S_train_df,
                                                                        random_state=i)
            X_test = X_test_df.to_numpy() 
            
            #our method
            start = time.time()
    
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train,y_train)
            
            clf = LogisticRegression()
            clf.fit(X_train,S_train)

            fair_reg = FairReg(reg, clf, B=B, K=K,  p=p, eps=eps, T=T, keep_history = False)
            fair_reg.fit(X_unlab, beta, L)
            
            end = time.time()
            time_hist.append(end-start)

            y_pred_base = reg.predict(X_test)
            y_pred_fair = fair_reg.predict(X_test)
            y_pred_prob_fair = fair_reg.pred_prob
            
            #r_X = np.square(y_pred_base[:, np.newaxis] - fair_reg.Q_L)
            r_X = np.square(y_test[:, np.newaxis] - fair_reg.Q_L)
            prob_risk.append(np.mean(np.sum(r_X*y_pred_prob_fair, axis=1)))
            
            mse_risk.append(mse(y_test, y_pred_fair))
            base_mse_risk.append(mse(y_test, y_pred_base))
            
            prob_unf = prob_unfairness(y_pred_prob_fair, S_test)
            DP = DP_unfairness(y_pred_fair, S_test, bins=fair_reg.Q_L)
            base_DP = DP_unfairness(y_pred_base, S_test, bins=fair_reg.Q_L)
            
            for s in range(K):
                avg_unf[s].append(prob_unf[s].mean()) #the average of prob. unfairness[s] in grid
                sum_unf[s].append(sum(prob_unf[s])) #the sum of prob. unfairness[s] in grid
                DP_unf[s].append(DP[s])
                base_DP_unf[s].append(base_DP[s])
            if print_details:    
                print ('-----   ', i,'/',num,': Our method: training completed; training time: ',end-start)
    
            #Agarwal et. al
            start = time.time()
            
            fair_reg_ADW = ADW.train_FairRegression(X_train_df, S_train_df, y_train_df, eps=eps[0],
                                    Theta=Theta,learner=learner,constraint="DP",loss="square")
            
            end = time.time()
            time_hist_ADW.append(end-start)
            
            eval_ADW = ADW.evaluate_FairModel(X_test_df, S_test, y_test, loss="square", 
                                          result=fair_reg_ADW['exp_grad_result'], Theta=Theta)
            
            loss_ADW.append(eval_ADW['weighted_loss'])

            for s in range(K):
                unf_ADW[s].append(eval_ADW['DP_disp'][s])
            
            if print_details:    
                print ('-----   ', i,'/',num,': ADW: training completed; training time: ',end-start)
        
        loss_ADW_all.append(np.mean(loss_ADW))
        
        prob_risk_all.append(np.mean(prob_risk))
        mse_risk_all.append(np.mean(mse_risk))
        base_mse_risk_all.append(np.mean(base_mse_risk))        
                
        for s in range(K):
            avg_all[s].append(np.mean(avg_unf[s]))
            sum_all[s].append(np.sum(sum_unf[s]))
            DP_all[s].append(np.mean(DP_unf[s]))
            unf_ADW_all[s].append(np.mean(unf_ADW[s]))
            base_DP_all[s].append(np.mean(base_DP_unf[s]))

        print ('---------------------------------------------------------')
        
    base_mse_risk = np.mean(base_mse_risk_all)
    for s in range(K):
        base_DP_all[s]=np.mean(base_DP_all[s])
     
    results = {'base_mse_risk':base_mse_risk,
              'base_DP':base_DP_all,
              'prob_risk':prob_risk_all,
              'mse_risk':mse_risk_all,
              'avg_unf':avg_all,
              'sum_unf':sum_all,
              'DP_unf':DP_all,
              'ADW_risk':loss_ADW_all,
              'ADW_unf':unf_ADW_all,
              'training_time_hist':time_hist,
              'ADW_training_time_hist':time_hist_ADW}
            
    return results
