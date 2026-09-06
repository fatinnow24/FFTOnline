import numpy as np
import matplotlib.pyplot as plt


def sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float) -> np.ndarray:
    raise NotImplementedError


def time_shift_sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float, n0: int) -> np.ndarray:
    raise NotImplementedError


def phase_change_sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float, phi0: float) -> np.ndarray:
    raise NotImplementedError


# -----------------------------
# 2) Utility functions
# -----------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two sequences of equal length."""
    return float(np.mean((a - b) ** 2))


def stem_plot(ax, n, x, label):
    """A nicer stem plot for discrete-time sequences."""
    markerline, stemlines, baseline = ax.stem(n, x, label=label)
    baseline.set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")


# -----------------------------
# 3) Main experiment
# -----------------------------
def main():
    # Base sinusoid parameters (you may change these to experiment)
    A = 1.0
    Omega0 = np.pi / 4
    phi = 0.0

    # Index range
    n = np.arange(-20, 21)  # -20, -19, ..., 20

    # Original signal
    x = None

    n0 = None  # integer time shift
    x_time = None

    # TODO: Compute the phase shift phi0_equiv that makes x_phase match x_time
    phi0_equiv = None  # TODO

    x_phase_equiv = None

    err_A = mse(x_time, x_phase_equiv)
    print("[Part A] MSE between time-shifted and equivalent phase-changed:", err_A)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    stem_plot(ax1, n, x, "original x[n]")
    stem_plot(ax1, n, x_time, f"time shift by n0={n0}")
    stem_plot(ax1, n, x_phase_equiv, f"phase change by phi0={phi0_equiv:.3f}")
    ax1.legend()
    fig1.tight_layout()

    
    phi0 = None  # an arbitrary phase change
    x_phase = None

    # Search over integer shifts to see if any time shift matches this phase change
    k_min, k_max = -12, 12
    best_k = None
    best_err = None

    for k in range(k_min, k_max + 1):
        x_time_k = time_shift_sinusoid(n, A, Omega0, phi, k)
        e = mse(x_time_k, x_phase)
        if (best_err is None) or (e < best_err):
            best_err = e
            best_k = k

    print(f"[Part B] Best matching integer shift in [{k_min},{k_max}] is k={best_k} with MSE={best_err}")

    x_time_best = time_shift_sinusoid(n, A, Omega0, phi, best_k)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    stem_plot(ax2, n, x_phase, f"phase change by phi0={phi0:.3f}")
    stem_plot(ax2, n, x_time_best, f"best time shift k={best_k}")
    ax2.legend()
    fig2.tight_layout()


if __name__ == "__main__":
    main()
