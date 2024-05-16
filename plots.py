import matplotlib.pyplot as plt
import numpy as np

def plot_distributions_compare(fair_model, base_model, X, S):
    
    S_val = sorted(S.unique())
    
    y_pred_fair = fair_model.predict(X)
    y_pred_base = base_model.predict(X)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,4))

    ax1.set_title("Fair model")
    for s in S_val:
        ax1.hist(y_pred_fair[S == s], label='s='+str(s), alpha=0.5, density=True, stacked=True, bins=20)
    ax1.legend()
        
    ax2.set_title("Base model")
    for s in S_val:
        ax2.hist(y_pred_base[S == s], label='s='+str(s), alpha=0.5, density=True, stacked=True, bins=20)
    ax2.legend()
    
    plt.show()
    
    
def plot_predictions_compare(fair_model, base_model, X, y):
    
    y_pred_fair = fair_model.predict(X)
    y_pred_base = base_model.predict(X)
    
        
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,4))

    ax1.set_title("Fair model")
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Predictions')
    ax1.scatter(y, y_pred_fair)
    ax1.plot(y, y, c = 'r')
    
    ax2.set_title("Base model")
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Predictions')
    ax2.scatter(y, y_pred_base)
    ax2.plot(y, y, c = 'r')
    
    plt.show()
    
        
def plot_distributions(model, X, S):   
    
    S_val = sorted(S.unique())
    
    y_pred = model.predict(X)

    plt.figure('1')
    for s in S_val:
        plt.hist(y_pred[S == s], label='s='+str(s), alpha=0.5, density=True, stacked=True, bins=20)
    plt.legend()
    plt.show()
    
        
def plot_predictions(model, X, y):
    
    y_pred = model.predict(X)

    plt.xlabel('Real')
    plt.ylabel('Predictions')
    plt.scatter(y, y_pred)
    plt.plot(y, y, c = 'r')
    plt.show()

#moving average function for smoothing the plots
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#plot unfairness history wrt number of iterations (takes the history from FairReg.history)
def plot_risk_history(risk_history_all, T, moving_av=1, dataset='law school'):
    
    risk_history = np.mean(risk_history_all, axis=0)
    risk_std = np.std(risk_history_all, axis=0)
    
    plt.xlabel("num of iterations")
    plt.ylabel("risk")
    plt.title(dataset)
    
    if moving_av==1:
        iter_ = np.arange(1, len(risk_history)+1)*(T/len(risk_history))
        plt.plot(iter_, risk_history, alpha=1, label="risk", c='r')
        plt.fill_between(iter_, risk_history-risk_std,risk_history+risk_std, alpha=.2, label='std', color='r', lw=0)
    elif moving_av>1:
        iter_ma = np.arange(moving_av, len(risk_history)+1)*(T/len(risk_history))
        plt.plot(iter_ma, moving_average(risk_history, moving_av), alpha=1, label="risk - MA", c='r')
        plt.fill_between(iter_ma, moving_average(risk_history-risk_std, moving_av),
                         moving_average(risk_history+risk_std, moving_av), alpha=.2, label='std', color='r', lw=0)
        
    plt.legend()
    
#plot unfairness history wrt number of iterations (takes the history from FairReg.history)
def plot_unfairness_history(unf_all, T, K, moving_av=1, unf_type='average unfairness in grid', dataset='law school'):
    
    colors = ['g', 'orange']
    plt.xlabel('num of iterations')
    plt.ylabel(unf_type)
    plt.title(dataset)
      
    for s in range(K):
        
        unf_history = np.mean(unf_all[s], axis=0)
        unf_std = np.std(unf_all[s], axis=0)
        
        if moving_av==1:
            iter_ = np.arange(1, len(unf_history)+1)*(T/len(unf_history))
            plt.plot(iter_, unf_history, alpha=1, label="unf : S="+str(s), c=colors[s])
            plt.fill_between(iter_, unf_history-unf_std,unf_history+unf_std, 
                             alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)
        elif moving_av>1:
            iter_ma = np.arange(moving_av, len(unf_history)+1)*(T/len(unf_history))
            plt.plot(iter_ma, moving_average(unf_history, moving_av),
                     alpha=1, label="unf - MA : S="+str(s),c=colors[s])
            plt.fill_between(iter_ma, moving_average(unf_history-unf_std, moving_av),
                             moving_average(unf_history+unf_std, moving_av), 
                             alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)

        plt.legend()
        
#plot unfairness vs risk wrt number of iterations        
def plot_unfairness_vs_risk(risk_history_all, unf_all, T, K, moving_av=1, dataset='law school', loglog=False):
    
    colors = ['g', 'orange']
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
        
    plt.xlabel('risk')
    plt.ylabel('average unfairness in grid')
    plt.title(dataset)
    
    risk_history = np.mean(risk_history_all, axis=0)
    risk_std = np.std(risk_history_all, axis=0)
      
    for s in range(K):
        
        unf_history = np.mean(unf_all[s], axis=0)
        unf_std = np.std(unf_all[s], axis=0)
        
        if moving_av==1:
            plt.plot(risk_history, unf_history, alpha=1, label="unf : S="+str(s), c=colors[s])
            plt.fill_between(risk_history, unf_history-unf_std,unf_history+unf_std, 
                             alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)
        elif moving_av>1:
            plt.plot(risk_history[moving_av-1:], moving_average(unf_history, moving_av),
                     alpha=1, label="avg unf - MA : S="+str(s),c=colors[s])
            plt.fill_between(risk_history[moving_av-1:], moving_average(unf_history-unf_std, moving_av),
                             moving_average(unf_history+unf_std, moving_av), 
                             alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)
        plt.legend()
    
#plot unfairness vs risk wrt pairs of epsilon thresholds           
def plot_risk_unf_wrt_eps(unf_all, risk_all, K=2, unf_type='average unfairness in grid', 
                          risk_type='probabilistic risk', dataset='communities and crime',
                          colors = ['g', 'orange'], legend_size=8, alpha=0.7, loglog=False):  
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
        
    plt.title(dataset)
    plt.xlabel(unf_type)
    plt.ylabel(risk_type)
    
    for s in range(K):
        plt.scatter(unf_all[s], risk_all, label='s='+str(s), alpha=alpha, color=colors[s])
        for i in range(len(risk_all)):
            plt.annotate('eps'+str(i+1), (unf_all[s][i], risk_all[i]), fontsize=8)
            
    plt.legend(prop={'size': legend_size})
    plt.show()
    

#plot different types of unfairness vs risk wrt pairs of epsilon thresholds      
def plot_risk_unf_compare(pairs_list, model_list, unf_type_list, risk_type_list,
                          markers_list=['o','s','x'], dataset='communities and crime', 
                          x_label = 'unfairness', y_label = 'risk',
                          S_list=[0], colors = [['tab:green'],['tab:orange']], legend_size=8, alpha=0.7, 
                          annotate=False, loglog=False, start_0=False):
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title(dataset)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for i, pair in enumerate(pairs_list):
        unf, risk = pair[0], pair[1]
        for s in S_list:
            
            LABEL = str(model_list[i])+' | '+str(unf_type_list[i])+' | '+str(risk_type_list[i])+' | s='+str(s)
            if model_list[i] == 'base':
                ALPHA=1
            else:
                ALPHA=alpha
                
            if isinstance(unf[s], dict):
                unf_=unf[s]['mean']
            else:
                unf_=unf[s]
            if isinstance(risk, dict):
                risk_=risk['mean']
            else:
                risk_=risk
                
            plt.plot(unf_, risk_, label=LABEL, marker=markers_list[i], linestyle='dashed', color=colors[i][s], alpha=ALPHA)
            if annotate:
                if model_list[i] != 'base':
                    for j in range(len(risk_)):
                        plt.annotate('eps'+str(j+1), (unf_[j], risk_[j]), fontsize=8)
            
    plt.legend(prop={'size': legend_size})
    if start_0:
        plt.xlim(left=0)
        plt.ylim(bottom=0)
    plt.show()
    
    
def plot_time_compare(hist_1, hist_2, eps_list, labels=['unaware-fair', 'ADW'], dataset= 'communities and crime',
                     loglog=False):
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title(dataset)
    plt.xlabel('epsilon')
    plt.ylabel('training time')
    
    k = int(len(hist_1)/len(eps_list))

    hist_1_mean = []
    hist_2_mean = []
    
    for i in range (0, len(eps_list)):
        hist_1_mean.append(np.mean(hist_1[k*i:k*i+3]))
        hist_2_mean.append(np.mean(hist_2[k*i:k*i+3]))
    
    plt.plot(eps_list, hist_1_mean, label=labels[0], marker='o', linestyle='dashed', color = 'tab:blue', alpha=0.9)
    for x, y in zip(eps_list, hist_1_mean):
        plt.text(x, y, f'{round(y,2)}', color='black', ha='center')
        
    plt.plot(eps_list, hist_2_mean, label=labels[1], marker='o', linestyle='dashed', color = 'tab:orange', alpha=0.9)
    for x, y in zip(eps_list, hist_2_mean):
        plt.text(x, y, f'{round(y,2)}', color='black', ha='center')
        
    plt.legend()
    plt.show()