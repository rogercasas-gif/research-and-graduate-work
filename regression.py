import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(7)
N_TEST = 10000


def sample(n, d, case):
    x = rng.uniform(-5, 5, size=(n, d))
    if case == "a":
        f = x.sum(axis=1)
    elif case == "b":
        f = 0.5 * (x ** 2).sum(axis=1)
    else:
        f = (x ** 2 / np.arange(1, d + 1)).sum(axis=1)
    return x, f + rng.standard_normal(n)


def features(x, fs):
    n, d = x.shape
    one = np.ones((n, 1))
    if fs == 1:
        return np.hstack([x, one])
    if fs == 2:
        return np.hstack([x, one, (x ** 2).sum(axis=1, keepdims=True)])
    i, j = np.triu_indices(d)
    return np.hstack([x[:, i] * x[:, j], one])


def n_features(d, fs):
    return {1: d + 1, 2: d + 2, 3: d * (d + 1) // 2 + 1}[fs]


def test_mse(n, d, case, fs):
    x, y = sample(n, d, case)
    w = np.linalg.lstsq(features(x, fs), y, rcond=None)[0]
    xt, yt = sample(N_TEST, d, case)
    return np.mean((features(xt, fs) @ w - yt) ** 2)


def limit(d, case, fs):
    v = 125 - (25 / 3) ** 2
    a = 1 / np.arange(1, d + 1)
    err = {("a", 3): 25 / 3 * d,
           ("b", 1): v * d / 4,
           ("c", 1): v * np.sum(a ** 2),
           ("c", 2): v * np.sum((a - a.mean()) ** 2)}
    return 1 + err.get((case, fs), 0.0)


def curve(d, case, fs, ns, reps):
    m = np.array([[test_mse(n, d, case, fs) for _ in range(reps)] for n in ns])
    return np.percentile(m, [25, 50, 75], axis=1)


colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
labels = {1: "1: linear", 2: r"2: linear + $\|x\|^2$", 3: "3: products $x_ix_j$"}
titles = {"a": r"(a) $\sum X^{(i)}$",
          "b": r"(b) $\frac{1}{2}\sum (X^{(i)})^2$",
          "c": r"(c) $\sum (X^{(i)})^2/i$"}


def plot_vs_n(dims, reps, filename):
    fig, axes = plt.subplots(len(dims), 3, figsize=(13, 4 * len(dims)))
    for r, d in enumerate(dims):
        for c, case in enumerate("abc"):
            ax = axes[r, c]
            ns = np.unique(np.geomspace(5, 30 * n_features(d, 3), 26).astype(int))
            for fs in (1, 2, 3):
                p = n_features(d, fs)
                q = curve(d, case, fs, ns, reps)
                ax.plot(ns, q[1], color=colors[fs], label=labels[fs])
                ax.fill_between(ns, q[0], q[2], color=colors[fs], alpha=0.2)
                ax.axhline(limit(d, case, fs), color=colors[fs], ls="--", lw=1)
                ax.axvline(p, color=colors[fs], ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(0.5, 1e5)
            ax.set_title(f"{titles[case]},  d = {d}")
            ax.set_xlabel("n")
            if c == 0:
                ax.set_ylabel("test MSE")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_vs_d(sizes, dims, reps, filename):
    fig, axes = plt.subplots(len(sizes), 3, figsize=(13, 4 * len(sizes)))
    for r, n in enumerate(sizes):
        for c, case in enumerate("abc"):
            ax = axes[r, c]
            for fs in (1, 2, 3):
                q = np.array([np.percentile([test_mse(n, d, case, fs) for _ in range(reps)],
                                            [25, 50, 75]) for d in dims]).T
                ax.plot(dims, q[1], "o-", ms=3, color=colors[fs], label=labels[fs])
                ax.fill_between(dims, q[0], q[2], color=colors[fs], alpha=0.2)
                ax.plot(dims, [limit(d, case, fs) for d in dims], color=colors[fs],
                        ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_ylim(0.5, 1e5)
            ax.set_title(f"{titles[case]},  n = {n}")
            ax.set_xlabel("d")
            if c == 0:
                ax.set_ylabel("test MSE")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_vs_n([5, 30], 20, "figures/fig1_vs_n.png")
    plot_vs_d([200, 2000], [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60], 10,
              "figures/fig2_vs_d.png")
