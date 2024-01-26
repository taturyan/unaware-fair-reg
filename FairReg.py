import numpy as np
from scipy.special import softmax
import collections
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class FairReg:
    def __init__(self, base_method, classifier, B, K, p, eps, T):
        self.base_method = base_method
        self.classifier = classifier
        self.B = B #bound on target
        self.K = K #number of sensitive attributes
        self.p = p #frequencies
        self.eps = eps #epsilon threshold
        self.T = T #number of iterations

    def stoch_grad(self, w, x):

        reg_pred = self.base_method.predict(x.reshape(1, -1))
        clf_prob = self.classifier.predict_proba(x.reshape(1, -1))
        tau_x_coef = np.zeros(clf_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_x_coef[:,i] = 1 - clf_prob[:,i]/p_i 

        r_x = np.square(reg_pred[:, np.newaxis] - self.Q_L)
        
        #computing the shtochastic gradient
        grad = np.zeros(2*self.K*(2*self.L+1))
        diff = (w[:self.K*(2*self.L+1)].copy() - w[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
        softmax_x = softmax(self.beta*(np.matmul(tau_x_coef,diff) - r_x), axis = 1)
        for i in range(self.K):
            grad[i*(2*self.L+1):(i+1)*(2*self.L+1)] = np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.eps[i]
            grad[(i+self.K)*(2*self.L+1):(i+1+self.K)*(2*self.L+1)] = -np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.eps[i]
        return grad
    
    def stoch_grad_reg(self, w, x, w_reg=[], mu_reg=[]):
        reg = 0
        for i in range(len(w_reg)):
            reg += mu_reg[i]*(w-w_reg[i])
        return self.stoch_grad(w, x) + reg

    def SGD(self, X, w_init, alpha, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_all = []
        for t in range(T):
            x = X[np.random.randint(len(X))]  
            w -= alpha * self.stoch_grad_reg(w, x, w_reg, mu_reg)
            w[w<0] = 0
            w_all.append(w)  

        return np.mean(w_all, axis=0)  
    
    def SGD_sc(self, X, w_init, mu, M, T, w_reg=[], mu_reg=[]):
        N1 = int(np.floor(T / (8 * M / mu)))
        K1 = int(np.floor(np.log(mu * T / (16 * M))))
        w = w_init
        for t in range(1, N1 + 1):
            w = self.SGD(X, w, 1 / (2 * M), int(4 * M / mu), w_reg, mu_reg)
        for k in range(1, K1 + 1):
            w = self.SGD(X, w, 1 / (2**k * M), int((2**(k + 2)) * M / mu), w_reg, mu_reg)
        return w

    def SGD3_sc(self, X, w_init, mu_init, M, T, sc = True):
        w = w_init
        mu = mu_init
        w_reg = []
        mu_reg = []
        if sc == False:
            w_reg.append(w)
            mu_reg.append(w)
        S1 = int(np.floor(np.log(M/mu)))
        for s in range(1, S1+1):
            w = self.SGD_sc(X, w, mu, 3*M, int(T/S1), w_reg, mu_reg)
            mu = 2*mu
            w_reg.append(w)
            mu_reg.append(mu)
        return w

    def SGD3(self, X, w_init, mu, M, T):
        return self.SGD3_sc(X, w_init, mu, M + mu, T, sc = False)

    def fit(self, X):
        
        sum_ps=0
        for p_s in self.p:
            sum_ps += (1-p_s)/p_s
        
        self.L = int(np.sqrt(self.T)) #discretization param
        self.Q_L = np.arange(-self.L, self.L + 1) * self.B / self.L #the grid
        self.beta = np.sqrt(self.T)/np.log(self.T) #temperature param in softmax
        self.M = self.beta*sum_ps #Lipschits constant
        self.mu = sum_ps/self.beta #strong-convexity param
        self.w_0 = np.zeros(2*self.K*(2*self.L+1)) #initial point
        
        
        self.w_est = self.SGD3(X, self.w_0, self.mu, self.M, self.T)

    def predict(self, X):
        
        reg_pred = self.base_method.predict(X)
        clf_prob = self.classifier.predict_proba(X)
        
        tau_X_coef = np.zeros(clf_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_X_coef[:,i] = 1 - clf_prob[:,i]/p_i 
        
        r_X = np.square(reg_pred[:, np.newaxis] - self.Q_L)
        
        diff = (self.w_est[:self.K*(2*self.L+1)].copy() - self.w_est[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
        
        pred_prob = softmax(self.beta*(np.matmul(tau_X_coef,diff) - r_X), axis = 1)
        self.pred_prob = pred_prob
        
        return (np.argmax(pred_prob, axis=1)-self.L)*self.B/self.L

