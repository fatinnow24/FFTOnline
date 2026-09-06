"""
CSE220 Online 2 (Step Response)

Instructions:
- Copy (or import) your completed Signal and LTI_System classes from Offline 1.
- Implement the TODO functions below.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt


# Paste/Import your Offline 1 implementations here
# from your_offline1_file import Signal, LTI_System

class Signal:
    def __init__(self, INF):
        # TODO: paste your Offline 1 implementation
        raise NotImplementedError

    def set_value_at_time(self, t, value):
        raise NotImplementedError

    def shift(self, k):
        raise NotImplementedError

    def add(self, other):
        raise NotImplementedError

    def multiply(self, scalar):
        raise NotImplementedError

    def plot(self, title="Discrete Signal"):
        raise NotImplementedError


class LTI_System:
    def __init__(self, impulse_response: Signal):
        # TODO: paste your Offline 1 implementation

    def linear_combination_of_impulses(self, input_signal: Signal):
        raise NotImplementedError

    def output(self, input_signal: Signal):
        raise NotImplementedError


def read_signal_from_file(filename: str, INF: int) -> Signal:
    sig = Signal(INF)
    with open(filename, "r", encoding="utf-8") as f:
        nstart, nend = map(int, f.readline().strip().split())
        vals = list(map(float, f.readline().strip().split()))
    assert len(vals) == (nend - nstart + 1)
    for i, v in enumerate(vals):
        sig.set_value_at_time(nstart + i, v)
    return sig


def first_difference(sig: Signal) -> Signal:
    """
    Returns Δsig[n] = sig[n] - sig[n-1] (assume outside range is 0).
    Must use Signal.shift/add/multiply.
    """
    # TODO
    raise NotImplementedError


def impulse_from_step_response(step_response: Signal) -> Signal:
    """
    Given s[n], compute h[n] = s[n] - s[n-1] (with s[-1]=0).
    Must use only Signal operations.
    """
    # TODO
    raise NotImplementedError


def output_using_step_response(x: Signal, step_response: Signal) -> Signal:
    """
    Compute y[n] using ONLY step response:
        y = (Δx * s)
    You must reuse your Offline 1 LTI_System machinery (linear combination of impulses).
    """
    # TODO
    raise NotImplementedError


# Main (demo workflow)
if __name__ == "__main__":
    # Choose INF large enough for your signals
    INF = 50

    # ---- Load provided files ----
    s = read_signal_from_file("step_response.txt", INF)
    x = read_signal_from_file("input_signal.txt", INF)

    # ---- Part 1: recover impulse response ----
    h = None

    s.plot("Step Response s[n]")
    h.plot("Recovered Impulse Response h[n] = s[n] - s[n-1]")

    # ---- Part 2: output using only step response ----
    dx = None
    y_s = None

    x.plot("Input x[n]")
    dx.plot("First Difference Δx[n]")
    y_s.plot("Output y_s[n] computed via step response")

    # ---- Part 3: verify with impulse-response method ----
    sys_h = None
    y_h = None
    y_h.plot("Output y_h[n] computed via impulse response")

    # Check if outputs match closely
    if np.allclose(y_s.values, y_h.values, atol=1e-6):
        print("Outputs match closely!")
    else:
        print("Outputs differ!")
