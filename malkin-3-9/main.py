import numpy as np
import matplotlib.pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, r_0:float,  label: str) -> None:
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(x, y, label=label)
    plt.axvline(x=r_0, color="red", linestyle="--", label=f"Yield Radius (r_0={r_0:.2f})")
    plt.axvline(x=-r_0, color="red", linestyle="--", label=f"Yield Radius (r_0={r_0:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1,1)
    plt.ylim(0)
    plt.title(label)
    plt.grid()
    plt.show()

def velocity_field(
    r: np.ndarray,
    R: float,
    Delta_P: float,
    mu: float,
    L: float,
    sigma_Y: float,
) -> tuple[np.ndarray, float]:
    a = Delta_P / (4 * mu * L)
    rabs = np.abs(r)
    r0 = (2 * L * sigma_Y) / Delta_P

    u = np.empty_like(r, dtype=float)

    # yielded region
    mask = rabs >= r0
    u[mask] = a * (R**2 - rabs[mask]**2) - (sigma_Y / mu) * (R - rabs[mask])

    # plug region: constant = u(r0)
    u_plug = a * (R**2 - r0**2) - (sigma_Y / mu) * (R - r0)
    u[~mask] = u_plug

    return u, r0

def shear_rate_field(
    r: np.ndarray,
    Delta_P: float,
    mu: float,
    L: float,
    sigma_Y: float,
) -> np.ndarray:
    r0 = (2 * L * sigma_Y) / Delta_P
    gamma_dot = np.zeros_like(r)
    for i, _ in enumerate(r):
        if abs(r[i]) < r0:
            gamma_dot[i] = 0.0
        else:
            gamma_dot[i] = abs(Delta_P / (2 * mu * L) * r[i]) - sigma_Y / mu
    return gamma_dot

def shear_stress_field(
    r: np.ndarray,
    Delta_P: float,
    mu: float,
    L: float,
    sigma_Y: float,
) -> np.ndarray:
    r0 = (2 * L * sigma_Y) / Delta_P
    tau = np.zeros_like(r)
    for i, _ in enumerate(r):
        if abs(r[i]) < r0:
            tau[i] = sigma_Y
        else:
            tau[i] = abs(Delta_P / (2 * L) * r[i])
    return tau

def heatmap_lengthwise(
    r: np.ndarray,
    field: np.ndarray,
    R: float,
    length: float,
    label: str,
    cmap: str = "viridis",
) -> None:
    """
    Creates a rectangular 'cut pipe' heatmap.
    x-axis: tube length (z)
    y-axis: radial position (r)
    color: field value
    """

    # number of points along tube axis
    nz = 400
    z = np.linspace(0, length, nz)

    # extend radial profile uniformly along z
    field_2d = np.tile(field[:, None], (1, nz))

    plt.figure(figsize=(10, 3), dpi=200)

    im = plt.imshow(
        field_2d,
        extent=[0, length, -R, R],
        aspect="auto",      # <- makes it rectangular
        origin="lower",
        cmap=cmap,
    )

    plt.colorbar(im, label=label)
    plt.xlabel("Tube Length (z)")
    plt.ylabel("Radius r")
    plt.title(f"{label} (Lengthwise Cut)")
    plt.tight_layout()
    plt.show()


def heatmap_cross_section(
    r: np.ndarray,
    field: np.ndarray,
    R: float,
    label: str,
    cmap: str = "viridis",
) -> None:
    """
    Creates a circular 'cross section' heatmap.
    x-axis: radial position (r)
    y-axis: angular position (theta)
    color: field value
    """

    # number of angular points
    ntheta = 400
    theta = np.linspace(0, 2 * np.pi, ntheta)

    # create 2D grid for cross section
    R_grid, Theta_grid = np.meshgrid(r, theta)

    # convert to Cartesian coordinates for plotting
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    # extend field values uniformly in angular direction
    field_2d = np.tile(field[:, None], (1, ntheta))

    plt.figure(figsize=(6, 6), dpi=200)

    im = plt.pcolormesh(X, Y, field_2d.T, shading='auto', cmap=cmap)

    plt.colorbar(im, label=label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{label} (Cross Section)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main() -> None:
    R = 1
    r = np.linspace(-R, R, 500)
    R: float = 1.0
    Delta_P: float = 1.0
    mu: float = 1.0
    L: float = 1.0
    sigma_Y: float = 0.2
    r_0 = (2 * L * sigma_Y) / Delta_P

    u,_ = velocity_field(r, R, Delta_P, mu, L, sigma_Y)
    gamma_dot = shear_rate_field(r, Delta_P, mu, L, sigma_Y)
    tau = shear_stress_field(r, Delta_P, mu, L, sigma_Y)
    plot(r, u, r_0, label="Velocity Field")
    plot(r, gamma_dot, r_0, label="Shear Rate Field")
    plot(r, tau, r_0, label="Shear Stress Field")

    heatmap_lengthwise(r, u, R, L, label="Velocity Field")
    heatmap_lengthwise(r, gamma_dot, R, L, label="Shear Rate Field")
    heatmap_lengthwise(r, tau, R, L, label="Shear Stress Field")

    heatmap_cross_section(r, u, R, label="Velocity Field")
    heatmap_cross_section(r, gamma_dot, R, label="Shear Rate Field")
    heatmap_cross_section(r, tau, R, label="Shear Stress Field")




if __name__ == "__main__":
    main()
