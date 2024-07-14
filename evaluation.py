import AgarwalDudikWu as ADW

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
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from FairReg import FairReg
from evaluation_measures import DP_unfairness, prob_unfairness, unfairness, prob_unfairness_summary, DP_unfairness_summary, prob_risk
from data_prep import get_lawschool_data, get_communities_data, get_adult_data, get_frequencies

def get_stats(dataset='lawschool', num=10, T=1000, eps = [0.01, 0.01], beta='auto', L='auto',
              TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=True):
   
    if dataset=='lawschool':
        X, S, y = get_lawschool_data()
        # normalizing data to [0,1]
        y = y/4 
        #scaling
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, test_size=1 - sample_size / (len(X)), random_state=42)
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='adult':
        X, S, y = get_adult_data()
        # normalizing data to [0,1]
        y = y/100
        #scaling
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, test_size=1 - sample_size / (len(X)), random_state=42)
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        B=1
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
    
    risk_history_all = []
    unf_all, base_DP_all = {}, {}
    for s in range(K):
        unf_all[s], base_DP_all[s] = [], []
 
    for i in range(1, num+1):
    
        print (i,'/',num, ': training...')

        X_train, X_, S_train, S_, y_train, y_ = train_test_split(X, S, y, test_size=TEST_SIZE, stratify=S, random_state=i)
        X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(X_, S_, y_, 
                                                                                 train_size = TRAIN_SIZE/(1-TEST_SIZE), stratify=S_,
                                                                                 random_state=i)

        reg = LinearRegression(fit_intercept=True)
        clf = LogisticRegression()
            
        reg.fit(X_train,y_train)
        clf.fit(X_train,S_train)

        fair_reg = FairReg(reg, clf, B=B, K=K,  p=p, eps=eps, T=T, keep_history = True)
        fair_reg.fit(X_unlab, beta, L)

        print ('training completed')
        print ('retrieving training history...')

        risk_history, unfairness_history = fair_reg.history(X_test, S_test, y_test, data_scaling=True, scaler=scaler)
        risk_history_all.append(risk_history)
                
        unf = DP_unfairness_summary(unfairness_history, fair_reg.K)

        for s in range(K):
            unf_all[s].append(unf[s])

        print ('training history retrieved')
        print ('====================================')
                
    results = {'risk':risk_history_all,
               'unf':unf_all}            
        
    return results
    
    
    
def get_risk_unf_wrt_eps(dataset, num, T, eps_list, print_details = True, beta='auto', L='auto',
                         alg={'base':'SGD3', 'method':'ACSA2'},
                         TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=True):
    
    if dataset=='lawschool':
        X, S, y = get_lawschool_data()
        #normalizing data to [0,1]
        y = y/4 
        #scaling
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, test_size=1 - sample_size / (len(X)), random_state=42)
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data()
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='adult':
        X, S, y = get_adult_data()
        # normalizing data to [0,1]
        y = y/100
        #scaling
        if data_scaling:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
            y = pd.Series(y_scaled.flatten(), index=y.index)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, test_size=1 - sample_size / (len(X)), random_state=42)
        B=1
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
    
    time_hist = []
    prob_risk_all = {'mean':[], 'std':[]}
    base_mse_risk, base_mse_risk_all = {}, []
    classif_score = []
    avg_prob_unf_all, unf_all, base_DP_unf, base_DP_unf_all = {}, {}, {}, {}
    for s in range(K):
        avg_prob_unf_all[s], unf_all[s], base_DP_unf[s], base_DP_unf_all[s] = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}, {}, []
 
    for k, eps in enumerate(eps_list):
        if print_details:
            print (k+1,'/',len(eps_list), ' : collecting statistics for eps='+str(eps))
        
        prob_risk_ = []
        avg_prob_unf, unf = {}, {}
        for s in range(K):
            avg_prob_unf[s], unf[s] = [], []
    
        for i in range(1, num+1):

            X_train, X_, S_train, S_, y_train, y_ = train_test_split(X, S, y, test_size=TEST_SIZE, stratify=S, random_state=i)
            X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(X_, S_, y_, 
                                                                                 train_size = TRAIN_SIZE/(1-TEST_SIZE), stratify=S_,
                                                                                 random_state=i)
            
            start = time.time()
            
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_train,y_train)
            
            clf = LogisticRegression()
            clf.fit(X_train,S_train)
            
            fair_reg = FairReg(reg, clf, B=B, K=K,  p=p, eps=eps, T=T, keep_history = False)
            fair_reg.fit(X_unlab, beta, L, alg)
            
            end = time.time()
            time_hist.append(end-start)
            
            classif_score.append(clf.score(X_test, S_test))
            
            y_pred_base = reg.predict(X_test)
            y_pred_fair = fair_reg.predict(X_test)
            y_pred_prob_fair = fair_reg.pred_prob
            
            #inverse scaling for evaluation
            if data_scaling:
                y_pred_base = scaler.inverse_transform(y_pred_base.reshape(-1, 1))[:,0]
                y_test = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:,0]
                grid = scaler.inverse_transform(fair_reg.Q_L.reshape(-1, 1))[:,0] 
            else:
                grid = fair_reg.Q_L
            
            prob_risk_.append(prob_risk(y_test, grid, y_pred_prob_fair))            
            base_mse_risk_all.append(mse(y_test, y_pred_base))
            
            prob_unf = prob_unfairness(y_pred_prob_fair, S_test)
            DP_unf = unfairness(y_pred_prob_fair, S_test)
            base_DP = DP_unfairness(y_pred_base, S_test, bins=grid)
            
            for s in range(K):
                avg_prob_unf[s].append(prob_unf[s].mean()) #the average of prob. unfairness[s] in grid
                unf[s].append(DP_unf[s])
                base_DP_unf_all[s].append(base_DP[s])
            if print_details:    
                print ('-----   ', i,'/',num,': training completed, statistics collected')
        
        prob_risk_all['mean'].append(np.mean(prob_risk_))
        prob_risk_all['std'].append(np.std(prob_risk_))
                
        for s in range(K):
            avg_prob_unf_all[s]['mean'].append(np.mean(avg_prob_unf[s]))
            avg_prob_unf_all[s]['std'].append(np.std(avg_prob_unf[s]))
            unf_all[s]['mean'].append(np.mean(unf[s]))
            unf_all[s]['std'].append(np.std(unf[s]))
            
        print ('---------------------------------------------------------')

    base_mse_risk['mean'] = np.mean(base_mse_risk_all)
    base_mse_risk['std'] = np.std(base_mse_risk_all)
    for s in range(K):
        base_DP_unf[s]['mean'] = np.mean(base_DP_unf_all[s])
        base_DP_unf[s]['std'] = np.std(base_DP_unf_all[s])
            
    results = {'risk':prob_risk_all,
               'unf':unf_all,
               'avg_prob_unf':avg_prob_unf_all,
               'base_risk':base_mse_risk,
               'base_DP':base_DP_unf,
               'training_time_hist':time_hist,
               'clf_score': np.mean(classif_score)}        
            
    return results

def get_stats_ADW(dataset, num, eps_list, print_details = True,
            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, partial_training=False):
    
    #geting the data
    if dataset=='lawschool':
        X_all, S_all, y_all = get_lawschool_data(as_df=True)
        sample_size = 2000 #we take only 2000 samples for comparison
        X, _, S, _, y ,_ = train_test_split(X_all, S_all, y_all, test_size=1 - sample_size / (len(X_all)), random_state=42)
        y = y/4 # normalizing data to [0,1]
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='communities':
        X, S, y = get_communities_data(as_df=True)
        B=1
        K=2
        p = get_frequencies(S)
    elif dataset=='adult':
        X_all, S_all, y_all = get_adult_data(as_df=True)
        sample_size = 2000 #we take only 2000 samples for comparison
        X, _, S, _, y ,_ = train_test_split(X_all, S_all, y_all, test_size=1 - sample_size / (len(X_all)), random_state=42)
        y = y/100 # normalizing data to [0,1]
        B=1
        K=2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')
    ####################################
        
    #initializations
    time_hist = []
    loss_all = []
    loss_std_all = []
    unf_all = {}
    unf_std_all = {}
    for s in range(K):
        unf_all[s] = []
        unf_std_all[s] = []
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
        
        loss = []
        unf = {}
        for s in range(K):
            unf[s] = []
     
        for i in range(1, num+1):

            X_df, X_test_df, S_df, S_test, y_df, y_test = train_test_split(X, S, y,
                                                                           test_size=TEST_SIZE, stratify=S, random_state=i)
            X_df.index, S_df.index, y_df.index = range(len(X_df)), range(len(S_df)), range(len(y_df))
            X_test_df.index, S_test.index, y_test.index = range(len(X_test_df)), range(len(S_test)), range(len(y_test))

            #additionally splitting into train and unlab according to our method
            if partial_training:
                X_train_df, X_unlab_df, S_train_df, S_unlab_df, y_train_df, y_unlab_df = train_test_split(X_df, S_df, y_df, 
                                                                            train_size = TRAIN_SIZE/(1-TEST_SIZE), stratify=S_df,
                                                                            random_state=i)
                X_train_df.index, S_train_df.index, y_train_df.index = range(len(X_train_df)),range(len(S_train_df)),range(len(y_train_df))
            else:
                X_train_df, S_train_df, y_train_df = X_df, S_df, y_df
    
            #Agarwal et. al
            start = time.time()
            
            fair_reg_ADW = ADW.train_FairRegression(X_train_df, S_train_df, y_train_df, eps=eps[0],
                                    Theta=Theta,learner=learner,constraint="DP",loss="square")
            
            end = time.time()
            time_hist.append(end-start)
            
            eval_ADW = ADW.evaluate_FairModel(X_test_df, S_test, y_test, loss="square", 
                                          result=fair_reg_ADW['exp_grad_result'], Theta=Theta)
            
            loss.append(eval_ADW['weighted_loss'])

            for s in range(K):
                unf[s].append(eval_ADW['DP_disp'][s])
            
            if print_details:    
                print ('-----   ', i,'/',num,': ADW: training completed; training time: ',end-start)
        
        loss_all.append(np.mean(loss))
        loss_std_all.append(np.std(loss)) 
                
        for s in range(K):
            unf_all[s].append(np.mean(unf[s]))
            unf_std_all[s].append(np.std(unf[s]))

        print ('---------------------------------------------------------')
        
     
    results = {'risk':loss_all,
               'risk_std':loss_std_all,
               'unf':unf_all,
               'unf_std':unf_std_all,
               'training_time_hist':time_hist}
            
    return results
