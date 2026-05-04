import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

theta_K = 1.0

def fn_levy(u):
    if u <= 0:
        return 0.0
    return (1.0 / (2.0 * np.sqrt(np.pi))) * u**(-1.5) * np.exp(-1.0 / (4.0 * u))

def G(theta):
    u = theta_K / theta
    return (theta_K / theta**2) * fn_levy(u)

def phi_reconstructed(t):
    def integrand(log_theta):
        theta = np.exp(log_theta)
        g = G(theta)
        if g <= 0:
            return 0.0
        return g * np.exp(-t / theta) * theta
    val, _ = quad(integrand, np.log(1e-6), np.log(1e6), limit=500,
                  epsabs=1e-12, epsrel=1e-10)
    return val

def phi_kww(t):
    return np.exp(-np.sqrt(t / theta_K))

t_plot   = np.logspace(-2, 2, 120)
t_sparse = np.logspace(-2, 2, 20)
kww_vals = phi_kww(t_plot)
rec_vals = np.array([phi_reconstructed(t) for t in t_sparse])

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(t_plot,   kww_vals, color="#d6604d", lw=2.5,
            label=r"$e^{-\sqrt{t/\theta_K}}$ (exact)")
ax.semilogx(t_sparse, rec_vals, "o", color="#1a9850", ms=7,
            mec="white", mew=0.5,
            label=r"$\int_0^\infty G(\theta)\,e^{-t/\theta}\,d\theta$")
ax.set_xlabel(r"$t / \theta_K$")
ax.set_ylabel(r"$\varphi(t)$")
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()

fig_path = "/home/gkerr/rheology-class/final_project/figures/kww_half.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
