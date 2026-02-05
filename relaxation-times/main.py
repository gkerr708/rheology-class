import numpy as np
from scipy.special import gammaln, gamma
from scipy.integrate import simpson
import matplotlib.pyplot as plt

#Substituting this into Eq. \ref{eq:relaxation_spectrum_kohlrausch} gives:
#\[
#G(\theta)
#=\frac{1}{\theta}\sum_{k=1}^\infty \frac{(-1)^{k+1}}{k!}\,
#\frac{\sin(\pi nk)}{\pi}\,\Gamma(1+nk)
#\left(\frac{\theta}{\theta_K}\right)^{nk},
#\qquad (0<n<1).
#\]
#We can attempt to show numerically that
#\[
#	\int_0^\infty G(\theta) e^{-t/\theta} d\theta
#	\approx e^{-(t/\theta_K)^n}.
#\]

def G_theta(theta, theta_K, n, max_k=100):
    """Compute the relaxation spectrum G(theta) for given theta, theta_K, and n."""
    G = np.zeros_like(theta)
    for k in range(1, max_k + 1):
        term = ((-1)**(k + 1) / gamma(k + 1)) * (np.sin(np.pi * n * k) / np.pi) * gamma(1 + n * k) * (theta / theta_K)**(n * k)
        G += term / theta
    return G

def relaxation_function(t, theta_K, n, max_k=100):
    """Compute the relaxation function by integrating G(theta) * exp(-t/theta)."""
    theta = np.logspace(-5, 5, 1000)  # Logarithmically spaced theta values
    G = G_theta(theta, theta_K, n, max_k=max_k)
    integrand = G * np.exp(-t / theta)
    return simpson(integrand, theta)


# Plot 
#plot relaxation function vs k 


max_k_values = np.arange(1, 10, 1)
t_values = np.logspace(-2, 2, 100)
theta_K = 1.0
n = 0.5
mean_relaxations = []
for max_k in max_k_values:
    relaxation_vals = [relaxation_function(t, theta_K, n, max_k=max_k) for t in t_values]
    mean_relaxations.append(np.mean(relaxation_vals))
plt.figure(figsize=(10, 6))
plt.plot(max_k_values, mean_relaxations, marker='o')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Max k value')
plt.ylabel('Mean Relaxation Function')
plt.title('Mean Relaxation Function vs Max k value')
plt.grid(True)
plt.show()









