# Metropolis-Hastings-Algorithm

For implementing Metropolis-Hastings algorithm simulated data were used with k=3 covariates and a constant term. A multivariate normal distribution was taken as a prior for ![formula](https://render.githubusercontent.com/render/math?math=\beta), N(0,![formula](https://render.githubusercontent.com/render/math?math=\Sigma)) where ![formula](https://render.githubusercontent.com/render/math?math=\Sigma) is a diagonal matrix with ![formula](https://render.githubusercontent.com/render/math?math=\frac{1}{\lambda}) on the
diagonal. After considering a number of values for ![formula](https://render.githubusercontent.com/render/math?math=\lambda), ![formula](https://render.githubusercontent.com/render/math?math=\lambda) = 1 was chosen. The posterior distribution was simulated using as proposal a multivariate normal distribution centered at the current update of and with a covariance matrix given by the inverse of Fisher information evaluated at the current update.

# Gibbs Sampler for binary data

For this analyse, we simulated N independent binary random variables where each yi comes from a Bernoulli distribution with probability of success ![formula](https://render.githubusercontent.com/render/math?math=\pi_{i}). 

