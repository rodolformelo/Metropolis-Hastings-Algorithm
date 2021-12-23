################### IMPORT ####################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, bernoulli
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
np.random.seed(0)


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

#Better perfomance to work in matrix form, avoid using loops
#PROOF: https://www.statlect.com/fundamentals-of-statistics/probit-model-maximum-likelihood
def probit_likelihood(beta, x, y):
    q = 2*y - 1
    eta = np.dot(x,beta)
    eta_q = np.multiply(q.T,eta)
    p = norm.cdf(eta_q)
    likehood = np.prod(p)
    return likehood

def beta_prior(beta, lambda0=10):
    mean = np.zeros(shape=(beta.shape[0],))
    cov = (1/lambda0) * np.identity(beta.shape[0])
    prior_distribution = multivariate_normal(mean=mean, cov=cov)
    return prior_distribution.pdf(beta)

def posterior(beta, x, y, lambda0=10):
    return beta_prior(beta, lambda0)*probit_likelihood(beta, x, y)

#Work in log form for numeric stability
def acceptance_probability(beta_new, beta_old, x, y, lambda0=10):
    posterior_new = posterior(beta_new, x, y, lambda0)
    posterior_old = posterior(beta_old, x, y, lambda0)
    logratio = np.log(posterior_new) - np.log(posterior_old)
    ratio = np.exp(logratio)
    return min(1, ratio)

#Fisher information for bernoulli Y with probit link
## Demidenko(2001) ("Computational aspects of probit model") & Graziani slides (Lecture 5, slide 7)
def Fisher_information_probit(beta,x):
    np.seterr(divide = 'ignore') ## avoid error while divide
    n,p = x.shape
    eta = np.dot(x,beta)
    numerator = norm.pdf(eta)**2
    denominator = (norm.cdf(eta))*(1.0001-norm.cdf(eta))
    value = np.divide(numerator, denominator)
    ## Set 1 when the value is 'inf', 1 = neutral multiplier
    ##value[value==np.inf] = 1 
    W = value * np.identity(n)
    I = x.T @ W @ x
    invI = np.linalg.inv(I)
    return invI

### Algorithm inverse fisher variance
def metropolsH_randomWalk_fisher(Y,X,lambda0 = 10, interations=5000, start=0, tau=1):
    n,p = X.shape
    np.random.seed(0)
    beta_tried = np.zeros(shape=(p,interations))
    beta_out = np.zeros(shape=(p,interations))
    beta_old = np.random.multivariate_normal(mean=start * np.ones((p,)), cov=np.identity((p))/lambda0)
    acpt = 0
    for i in range(interations):
      covariance = Fisher_information_probit(beta_old,X)
      beta_new = np.random.multivariate_normal(mean=beta_old, cov= tau * covariance) #either simulate a multiv normal with mean = beta or sum beta_old with mean zero
      beta_tried[:,i] = beta_new.T
      u = np.random.rand()
      alpha = acceptance_probability(beta_new, beta_old, X, Y, lambda0)
      if u < alpha:
          beta_old = beta_new
          acpt+=1   
      beta_out[:,i] = beta_old.T
    acpt_rate = acpt/interations
    return beta_out, beta_tried, acpt_rate


################### RUN THE MODEL #############################################
interations = 10000
### Change tau and lambda0 to check the better value -- Tau matter more than lambda0
MLE = np.array([0.7017, 0.6756, 0.4959, 0.4551])
### Start=0 also give us very good results
beta, beta_tried, acpt_rate = metropolsH_randomWalk_fisher(Y=Y,X=X, interations=interations, tau=0.7, lambda0=1, start=MLE) 
print(f"The acceptance rate was {np.round(acpt_rate*100,4)}% \n")
for n in range(beta.T[0].shape[0]):
    true = float(beta_true[n])
    m = np.around(np.mean(beta[n,:]),4)
    stdev = np.around(np.std(beta[n,:]),4)
    print(f"Beta({n}) mean prediction: {m}, 95% CI: ({round(m-1.96*stdev,4)},{round(m+1.96*stdev,4)}), true_beta: {true} \n")
    
    

################### DIAGNOSTICS ###############################################

#Plots
### Graphic Markov Chain
for n in range(beta.shape[0]):
    print(f"\n BETA({n}) \n")
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta[n,100:].T, color="black") 
    plt.axhline(beta_true[n], color='r', linestyle='-')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['Markov Chain','True Value'], fontsize=12, loc=1)
    plt.show()

### Histogram Markov Chain
for n in range(beta.shape[0]):
    print(f"\n BETA({n}) \n")
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.hist(beta[n,100:], bins=25, color="gray")
    plt.axvline(beta_true[n], color='r', linestyle='-')
    plt.axvline(np.mean(beta[n,:]), color='green', linestyle='-')
    plt.axvline(MLE[n], color='blue', linestyle='-')
    plt.xlabel('Density', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['True value', 'Markov mean', 'MLE','Density'], fontsize=12, loc=1)
    plt.show()

### Convergence to true value
for n in range(beta.shape[0]):
    beta_avg_cum = np.zeros(shape=(interations,))
    for t in range (interations):
        beta_avg_cum[t] = np.sum(beta[n,:t+1])/(t+1)
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
for n in range(beta.shape[0]):
    beta_avg_cum = np.zeros(shape=(interations,))
    for t in range (interations):
        beta_avg_cum[t] = np.sum(beta[n,:t+1])/(t+1)
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta[n,:].T, color="black")
    plt.plot(beta_avg_cum.T, color="blue")
    plt.axhline(beta_true[n], color='r', linestyle='-')
    plt.xlabel('Interations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['Markov Chain','Cumulative average','True value'], fontsize=12, loc=1)
    plt.show()
    
### Graphic to check beta tried and beta markov chain
for n in range(beta.shape[0]):
    bot = 5000
    up = 6000
    r = [i for i in range(up - bot)]
    print(f"\n BETA({n}) \n")
    plt.figure(figsize=(12,8))
    plt.title(f'Beta ({n})')
    plt.plot(beta[n,bot:up].T, color="black") 
    plt.scatter( r, beta_tried[n,bot:up].T, color="red", marker='*') 
    plt.axhline(beta_true[n], color='blue', linestyle='-')
    plt.xlabel('Interations', fontsize=12)
    plt.ylabel(f'Beta ({n})', fontsize=12)
    plt.legend(['Markov Chain', 'True value', 'Beta suggested'], fontsize=12, loc=1)
    plt.show()
    
#ACF values
for n in range(beta.shape[0]):
    burn_in = 5000
    acfunc = acf(beta[n,burn_in:].T, nlags=10, fft=False)
    print(f"The ACF matrix for Beta{n} is {acfunc}\n")

#ACF plots 
for n in range(beta.shape[0]):
    plt.rc("figure", figsize=(12,8))
    burn_in = 5000
    print(f"\n The ACF plot for Beta{n} is: \n")
    plot_acf(beta[n,burn_in:].T, lags=30, title = f'Auotocorrelation Beta {n}')
    plt.show()
    
#ACF plots -5 in five
for n in range(beta.shape[0]):
    plt.rc("figure", figsize=(12,8))
    burn_in = 5000
    print(f"\n The ACF plot for Beta{n} is: \n")
    plot_acf(beta[n,burn_in::5].T, lags=20, title = f'Auotocorrelation Beta {n} - Blocks of 5')
    plt.show()


################### BULT-IN  ##################################################
## MLE estimator
model = Probit(Y, X.astype(float))
probit_model = model.fit()
print("MLE estimate: \n")
print(probit_model.summary())




