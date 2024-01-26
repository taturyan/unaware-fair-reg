import matplotlib.pyplot as plt

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

