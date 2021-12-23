################### IMPORT ####################################################
from scipy.stats import norm, bernoulli, truncnorm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


################### SIMULATED DATA ############################################
# simulated data
np.random.seed(0)
n = 200 # number of observations
k = 3 # number of explanatory variables

X = np.ones((n,k+1))
X[:,1:] = np.random.normal(loc=0,scale=1,size=k*n).reshape(n,k)
beta_true = np.matrix( [0.8,0.7,0.4,0.6]).T
sigma_true = 0.1
eta = np.dot(X,beta_true) + np.random.normal(loc=0, scale = sigma_true, size=(n,1))
Y =  bernoulli.rvs(p = norm.cdf(eta))

################### ALGORITHM #################################################


def gibbs_sampler_informative(X,Y, inter=1000, lambda_0=10, start=0):
    np.random.seed(0)
    n,p = X.shape
    V_0 = lambda_0*np.diag(np.ones(p))
    invV_0 = la.inv(V_0)
    Cov = la.inv(invV_0 + X.T@X)
    beta_out = np.zeros(shape=(p,inter))
    beta_0 = start * np.ones((1,p))
    beta = beta_0
    Z = np.zeros((n,1))
    for i in range(2,inter):
        mu_z = np.dot(X,beta.T)
        Z[Y==0] = truncnorm.rvs(loc = mu_z[Y==0], scale=1, a= -np.inf, b= -mu_z[Y==0])
        Z[Y==1] = truncnorm.rvs(loc = mu_z[Y==1], scale=1, a= -mu_z[Y==1], b= np.inf)
        M = Cov @ (invV_0@beta_0.T + X.T@Z)
        beta = np.random.multivariate_normal(mean=M.T[0], cov= Cov).reshape(1,p)
        beta_out[:,i] = beta
    return beta_out

def gibbs_sampler_non_informative(X,Y, inter=1000, start=0):
    np.random.seed(0)
    n,p = X.shape
    beta_out = np.zeros(shape=(p,inter))
    beta_0 = start * np.ones((1,p))
    beta = beta_0
    Z = np.zeros((n,1)) #<- rep(0, N)
    Cov = la.inv(X.T@X)
    for i in range(2,inter):
        mu_z = np.dot(X,beta.T)
        Z[Y==0] = truncnorm.rvs(loc = mu_z[Y==0], scale=1, a= -np.inf, b= -mu_z[Y==0])
        Z[Y==1] = truncnorm.rvs(loc = mu_z[Y==1], scale=1, a= -mu_z[Y==1], b= np.inf)
        M = Cov @ X.T@Z
        beta = np.random.multivariate_normal(mean=M.T[0], cov= Cov).reshape(1,p)
        beta_out[:,i] = beta
    return beta_out


################### RUN THE MODEL #############################################
MLE = np.array([0.7017, 0.6756, 0.4959, 0.4551])
interations = 5000    
beta_i = gibbs_sampler_informative(X,Y, inter=interations, lambda_0=10, start=MLE)
for n in range(beta_i.T[0].shape[0]):
    true = float(beta_true[n])
    m = np.around(np.mean(beta_i[n,:]),4)
    stdev = np.around(np.std(beta_i[n,:]),4)
    print(f"Beta({n}) mean prediction: {m}, 95% CI: ({round(m-1.96*stdev,4)},{round(m+1.96*stdev,4)}), true_beta: {true} \n")

beta_ni = gibbs_sampler_non_informative(X,Y, inter=interations, start=0)
for n in range(beta_ni.T[0].shape[0]):
    true = float(beta_true[n])
    m = np.around(np.mean(beta_ni[n,:]),4)
    stdev = np.around(np.std(beta_i[n,:]),4)
    print(f"Beta({n}) mean prediction: {m}, 95% CI: ({round(m-1.96*stdev,4)},{round(m+1.96*stdev,4)}), true_beta: {true} \n")


################### DIAGNOSTICS ###############################################

#Plots
### Graphic Markov Chain
for n in range(beta_i.shape[0]):
    print(f"\n BETA({n}) \n")
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta_i[n,:].T, color="black") 
    plt.axhline(beta_true[n], color='r', linestyle='-')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['Markov Chain','True Value'], fontsize=12, loc=1)
    plt.show()

### Histogram Markov Chain
for n in range(beta_i.shape[0]):
    print(f"\n BETA({n}) \n")
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.hist(beta_i[n,:], bins=25, color="gray")
    plt.axvline(beta_true[n], color='r', linestyle='-')
    plt.axvline(np.mean(beta_i[n,:]), color='green', linestyle='-')
    plt.axvline(MLE[n], color='blue', linestyle='-')
    plt.xlabel('Density', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['True value', 'Markov mean', 'MLE','Density'], fontsize=12, loc=1)
    plt.show()

### Convergence to true value
for n in range(beta_i.shape[0]):
    beta_avg_cum = np.zeros(shape=(interations,))
    for t in range (interations):
        beta_avg_cum[t] = np.sum(beta_i[n,:t+1])/(t+1)
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta_avg_cum.T, color="black") 
    plt.axhline(beta_true[n], color='r', linestyle='-')
    plt.xlabel('Interations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.ylim(0.25,1)
    plt.legend(['Cumulative Average','True value'], fontsize=12, loc=1)
    plt.show()

 
## All Graphics toghether
for n in range(beta_i.shape[0]):
    beta_avg_cum = np.zeros(shape=(interations,))
    for t in range (interations):
        beta_avg_cum[t] = np.sum(beta_i[n,:t+1])/(t+1)
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta_i[n,:].T, color="black")
    plt.plot(beta_avg_cum.T, color="blue")
    plt.axhline(beta_true[n], color='r', linestyle='-')
    plt.xlabel('Interations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['Markov Chain','Cumulative average','True value'], fontsize=12, loc=1)
    plt.show()
    
    
#ACF values
for n in range(beta_i.shape[0]):
    burn_in = 2000
    acfunc = acf(beta_i[n,burn_in:].T, nlags=10, fft=False)
    print(f"The ACF matrix for Beta{n} is {acfunc}\n")

#ACF plots 
for n in range(beta_i.shape[0]):
    plt.rc("figure", figsize=(12,8))
    burn_in = 2000
    print(f"\n The ACF plot for Beta{n} is: \n")
    plot_acf(beta_i[n,burn_in:].T, lags=30, title = f'Auotocorrelation Beta {n}')
    plt.show()
    
#ACF plots -5 in five
for n in range(beta_i.shape[0]):
    plt.rc("figure", figsize=(12,8))
    burn_in = 2000
    print(f"\n The ACF plot for Beta{n} is: \n")
    plot_acf(beta_i[n,burn_in::5].T, lags=20, title = f'Auotocorrelation Beta {n} - Blocks of 5')
    plt.show()


################### BULT-IN  ##################################################
## MLE estimator
model = Probit(Y, X.astype(float))
probit_model = model.fit()
print("MLE estimate: \n")
print(probit_model.summary())
