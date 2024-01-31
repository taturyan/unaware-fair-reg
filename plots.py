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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_risk_history(risk_history_all, T, moving_av=1, dataset='law school'):
    
    risk_history = np.mean(risk_history_all, axis=0)
    risk_std = np.std(risk_history_all, axis=0)
    
    plt.xlabel("num of iterations")
    plt.ylabel("risk")
    plt.title(dataset)
    
    iter_ = np.arange(1, len(risk_history)+1)*(T/len(risk_history))
    plt.plot(iter_, risk_history, alpha=.5, label="risk", c='r')
    plt.fill_between(iter_, risk_history-risk_std,risk_history+risk_std, alpha=.2, color='r', lw=0)
    
    if moving_av>1:
        iter_ma = np.arange(moving_av, len(risk_history)+1)*(T/len(risk_history))
        plt.plot(iter_ma, moving_average(risk_history, moving_av), alpha=1, linestyle='dotted', label="risk - MA", c='r')
        
    plt.legend()
    
    
def plot_unfairness_history(unf_all, unf_std_all, T, K, moving_av=1, dataset='law school'):
    
    colors = ['g', 'orange']
    plt.xlabel('num of iterations')
    plt.ylabel('average unfairness in grid')
    plt.title(dataset)
      
    for s in range(K):
        
        unf_history = np.mean(unf_all[s], axis=0)
        unf_std = np.mean(unf_std_all[s], axis=0)
        
        iter_ = np.arange(1, len(unf_history)+1)*(T/len(unf_history))
        plt.plot(iter_, unf_history, alpha=.5, label="avg unf : S="+str(s), c=colors[s])
        plt.fill_between(iter_, unf_history-unf_std,unf_history+unf_std, 
                         alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)

        if moving_av>1:
            iter_ma = np.arange(moving_av, len(unf_history)+1)*(T/len(unf_history))
            plt.plot(iter_ma, moving_average(unf_history, moving_av),
                     alpha=1, linestyle='dotted', label="avg unf - MA : S="+str(s),c=colors[s])

        plt.legend()
        
        
def plot_unfairness_vs_risk(risk_history_all, unf_all, unf_std_all, T, K, moving_av=1, dataset='law school'):
    
    colors = ['g', 'orange']
    plt.xlabel('risk')
    plt.ylabel('average unfairness in grid')
    plt.title(dataset)
    
    risk_history = np.mean(risk_history_all, axis=0)
    risk_std = np.std(risk_history_all, axis=0)
      
    for s in range(K):
        
        unf_history = np.mean(unf_all[s], axis=0)
        unf_std = np.mean(unf_std_all[s], axis=0)
        
        plt.plot(risk_history, unf_history, alpha=.5, label="avg unf : S="+str(s), c=colors[s])
        plt.fill_between(risk_history, unf_history-unf_std,unf_history+unf_std, 
                         alpha=.2, label="std - S="+str(s), color=colors[s], lw=0)

        if moving_av>1:
            plt.plot(risk_history[moving_av-1:], moving_average(unf_history, moving_av),
                     alpha=1, linestyle='dotted', label="avg unf - MA : S="+str(s),c=colors[s])

        plt.legend()
    