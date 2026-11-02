import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(7)


def cube(n, d):
    return rng.uniform(-1, 1, size=(n, d))


def ball(n, d):
    z = rng.standard_normal((n, d))
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    r = rng.uniform(0, 1, size=(n, 1)) ** (1 / d)
    return r * z


def cross_polytope(n, d):
    e = rng.exponential(1, size=(n, d + 1))
    x = e[:, :d] / e.sum(axis=1, keepdims=True)
    s = rng.choice([-1, 1], size=(n, d))
    return s * x


def project(x):
    d = x.shape[1]
    g = rng.standard_normal((d, 2))
    q, _ = np.linalg.qr(g)
    y = x @ q
    y = y - y.mean(axis=0)
    return y / y[:, 0].std()


samplers = {"Cube": cube, "Ball": ball, "Cross polytope": cross_polytope}


def check(n=100000, d=5):
    x = cube(n, d)
    print("cube ", np.abs(x).max(), np.mean(x[:, 0] ** 2), 1 / 3)
    x = ball(n, d)
    print("ball ", np.linalg.norm(x, axis=1).max(), np.mean(x[:, 0] ** 2), 1 / (d + 2))
    x = cross_polytope(n, d)
    print("cross", np.abs(x).sum(axis=1).max(), np.mean(x[:, 0] ** 2), 2 / ((d + 1) * (d + 2)))


def plot_grid(n, dims, filename):
    size = 6 if n <= 500 else (2 if n <= 5000 else 0.5)
    alpha = 0.8 if n <= 500 else 0.4
    fig, axes = plt.subplots(3, len(dims), figsize=(2.4 * len(dims), 7.2),
                             sharex=True, sharey=True)
    for i, (name, f) in enumerate(samplers.items()):
        for j, d in enumerate(dims):
            y = project(f(n, d))
            ax = axes[i, j]
            ax.scatter(y[:, 0], y[:, 1], s=size, alpha=alpha, color="black",
                       linewidths=0, rasterized=True)
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"d = {d}")
            if j == 0:
                ax.set_ylabel(name)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    check()
    plot_grid(3000, [2, 3, 5, 10, 50, 1000], "figures/fig1_n3000.png")
    plot_grid(100, [2, 3, 10, 1000], "figures/fig2_n100.png")
    plot_grid(50000, [3, 5, 10, 20, 100], "figures/fig3_n50000.png")
