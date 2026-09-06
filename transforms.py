"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


NTT_MODULUS = 998244353
NTT_PRIMITIVE_ROOT = 3


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    if n<=1:
        return 1
    power = 1
    while (power<n):
        power*=2
    return power


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        return X

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        X = np.asarray(spectrum, dtype=np.complex128)
        N = len(X)
        x = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                x[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)
        return x / N


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        X = np.asarray(x, dtype = np.complex128).copy()
        N = len(x)
        if N < 1 or (N & (N-1)) != 0:
            raise ValueError("N must be a power of two")
        j = 0
        for i in range(1,N):
            bit = N >> 1
            while (j & bit):
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                X[i], X[j] = X[j], X[i]
        size = 2
        while (size<=N):
            half = size//2
            stage_twiddle = np.exp(-2j * np.pi / size)
            for start in range(0,N,size):
                W = 1
                for k in range(half):
                    g = X[start + k]
                    h = W * X[start + k + half]
                    
                    X[start + k] = g + h
                    X[start + k + half] = g - h
                    
                    W *= stage_twiddle
            size *= 2
        return X
    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        if N < 1 or (N & (N-1)) != 0:
            raise ValueError("N must be a power of two")
        conj_spectrum = np.conj(spectrum)
        transformed = self.transform(conj_spectrum)
        result = np.conj(transformed) / N
        return result
    def transform_dif(self, x):
        """Forward FFT using iterative radix-2 DIF FFT."""
        X = np.asarray(x, dtype=np.complex128).copy()
        N = len(x)

        if N < 1 or (N & (N - 1)) != 0:
            raise ValueError("N must be a power of two")

        size = N

        while size >= 2:
            half = size // 2
            stage_twiddle = np.exp(-2j * np.pi / size)

            for start in range(0, N, size):
                W = 1

                for k in range(half):
                    g = X[start + k]
                    h = X[start + k + half]

                    X[start + k] = g + h
                    X[start + k + half] = (g - h) * W

                    W *= stage_twiddle

            size //= 2

        # Bit-reverse the output
        j = 0

        for i in range(1, N):
            bit = N >> 1

            while (j & bit):
                j ^= bit
                bit >>= 1

            j ^= bit

            if i < j:
                X[i], X[j] = X[j], X[i]

        return X

class NTTTransformer:
    """Exact radix-2 number theoretic transform modulo 998244353."""

    name = "ntt"
    modulus = NTT_MODULUS
    primitive_root = NTT_PRIMITIVE_ROOT

    def _transform(self, values, inverse):
        values = np.asarray(values, dtype=np.int64).copy()
        N = len(values)
        if N < 1 or (N & (N - 1)) != 0:
            raise ValueError("N must be a power of two")
        values %= self.modulus

        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                values[i], values[j] = values[j], values[i]

        size = 2
        while size <= N:
            root = pow(self.primitive_root,
                       (self.modulus - 1) // size, self.modulus)
            if inverse:
                root = pow(root, self.modulus - 2, self.modulus)
            half = size // 2
            for start in range(0, N, size):
                twiddle = 1
                for offset in range(half):
                    even = int(values[start + offset])
                    odd = (twiddle * int(values[start + offset + half])) % self.modulus
                    values[start + offset] = (even + odd) % self.modulus
                    values[start + offset + half] = (even - odd) % self.modulus
                    twiddle = (twiddle * root) % self.modulus
            size <<= 1

        if inverse:
            scale = pow(N, self.modulus - 2, self.modulus)
            values = (values * scale) % self.modulus
        return values.astype(np.int64)

    def transform(self, x):
        """Return the forward NTT of a power-of-two length array."""
        return self._transform(x, inverse=False)

    def inverse(self, spectrum):
        """Return the inverse NTT of a power-of-two length spectrum."""
        return self._transform(spectrum, inverse=True)


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        x = np.asarray(x, dtype = np.complex128)
        N = len(x)
        if(N==0):
            return x.copy()
        if(N==1):
            return x.copy()
        if (N & (N-1))==0:
            return super().transform(x)
        M = next_power_of_two(2*N -1)
        n = np.arange(N)
        chirp = np.exp(-1j * np.pi * n * n/N)
        a = np.zeros(M, dtype = np.complex128)
        a[:N] = x * chirp
        b = np.zeros(M, dtype = np.complex128)
        positive_chirp = np.exp(1j * np.pi * n * n/N)
        b[:N] = positive_chirp
        for i in range(1,N):
            b[M-i] = positive_chirp[i]
        A = super().transform(a)
        B = super().transform(b)
        
        C = A*B
        c = super().inverse(C)
        
        X = c[:N] * chirp
        return X

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        if(N==0):
            return np.array([], dtype=np.complex128)
        conj_spectrum = np.conj(spectrum)
        transformed = self.transform(conj_spectrum)
        result = np.conj(transformed) / N
        return result


class DTFTAnalyzer:
    """
    Numerical DTFT analyzer for finite-length discrete-time signals.

    The DTFT is evaluated directly from

        X(e^jw) = sum_n x[n] exp(-j*w*n)

    This class does not use FFT/DFT routines internally.
    """

    def __init__(self, n=None):
        """
        Optionally specify the sample indices n.

        If n is None, transform() assumes:
            n = [0, 1, ..., N-1]
        """
        self.n = None if n is None else np.asarray(n, dtype=np.int64).copy()
        self.name = "dtft"

    def transform(self, x, omega):
        """
        Compute the DTFT of x at the supplied angular frequencies omega.

        Parameters
        ----------
        x : array-like
            Discrete-time signal.
        omega : array-like or scalar
            Angular frequencies in radians/sample.

        Returns
        -------
        X : complex ndarray or complex scalar
            DTFT values X(e^jw).
        """
        x = np.asarray(x, dtype=np.complex128)

        if x.ndim != 1:
            raise ValueError("x must be a one-dimensional signal")

        N = len(x)

        if N < 1:
            raise ValueError("x must contain at least one sample")

        if self.n is None:
            n = np.arange(N)
        else:
            n = self.n

            if len(n) != N:
                raise ValueError("n and x must have the same length")

        omega = np.asarray(omega, dtype=np.float64)

        scalar_input = omega.ndim == 0
        omega = np.atleast_1d(omega)

        X = np.zeros(len(omega), dtype=np.complex128)

        for k in range(len(omega)):
            for i in range(N):
                X[k] += x[i] * np.exp(-1j * omega[k] * n[i])

        if scalar_input:
            return X[0]

        return X

    def frequency_axis(self, num_points=1024, centered=True):
        """
        Generate an angular-frequency axis for DTFT evaluation.

        centered=True:
            omega ranges from -pi to pi.

        centered=False:
            omega ranges from 0 to 2*pi.
        """
        if num_points < 1:
            raise ValueError("num_points must be positive")

        if centered:
            return np.linspace(-np.pi, np.pi, num_points)
        else:
            return np.linspace(0.0, 2.0 * np.pi, num_points)

    def magnitude(self, X):
        """
        Return the magnitude spectrum |X(e^jw)|.
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.abs(X)

    def phase(self, X):
        """
        Return the phase spectrum angle(X(e^jw)).
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.angle(X)

    def phase_unwrapped(self, X):
        """
        Return the unwrapped phase spectrum.
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.unwrap(np.angle(X))

    def real_part(self, X):
        """
        Return the real part of the DTFT.
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.real(X)

    def imaginary_part(self, X):
        """
        Return the imaginary part of the DTFT.
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.imag(X)

    def power_spectrum(self, X):
        """
        Return the power spectrum |X(e^jw)|^2.
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.abs(X) ** 2

    def energy_spectrum(self, X):
        """
        Return the squared magnitude spectrum.

        This is equivalent to power_spectrum().
        """
        X = np.asarray(X, dtype=np.complex128)
        return np.abs(X) ** 2

    def dtft_from_frequency(self, x, frequency, fs=1.0):
        """
        Compute DTFT using ordinary frequency instead of angular frequency.

        frequency is in cycles per second (Hz).
        fs is the sampling frequency in Hz.

        omega = 2*pi*f/fs
        """
        frequency = np.asarray(frequency, dtype=np.float64)

        if fs <= 0:
            raise ValueError("fs must be positive")

        omega = 2.0 * np.pi * frequency / fs

        return self.transform(x, omega)

    def frequency_from_omega(self, omega, fs=1.0):
        """
        Convert angular frequency omega (rad/sample)
        to ordinary frequency f (Hz).
        """
        if fs <= 0:
            raise ValueError("fs must be positive")

        omega = np.asarray(omega, dtype=np.float64)

        return omega * fs / (2.0 * np.pi)

    def omega_from_frequency(self, frequency, fs=1.0):
        """
        Convert ordinary frequency f (Hz)
        to angular frequency omega (rad/sample).
        """
        if fs <= 0:
            raise ValueError("fs must be positive")

        frequency = np.asarray(frequency, dtype=np.float64)

        return 2.0 * np.pi * frequency / fs

    def magnitude_phase(self, x, omega):
        """
        Compute the DTFT and return magnitude and phase.
        """
        X = self.transform(x, omega)

        magnitude = np.abs(X)
        phase = np.angle(X)

        return X, magnitude, phase

    def real_imaginary(self, x, omega):
        """
        Compute the DTFT and return real and imaginary parts.
        """
        X = self.transform(x, omega)

        return X, np.real(X), np.imag(X)

    def reconstruct(self, X, omega, n):
        """
        Numerically reconstruct x[n] from DTFT samples using

            x[n] = (1 / 2*pi) integral X(e^jw)e^(jwn) dw

        The integral is approximated using the trapezoidal rule.

        X and omega must describe the same frequency grid.
        """
        X = np.asarray(X, dtype=np.complex128)
        omega = np.asarray(omega, dtype=np.float64)
        n = np.asarray(n, dtype=np.int64)

        if X.ndim != 1 or omega.ndim != 1:
            raise ValueError("X and omega must be one-dimensional")

        if len(X) != len(omega):
            raise ValueError("X and omega must have the same length")

        if len(omega) < 2:
            raise ValueError("At least two frequency points are required")

        result = np.zeros(len(n), dtype=np.complex128)

        for k in range(len(n)):
            integrand = X * np.exp(1j * omega * n[k])
            result[k] = np.trapezoid(integrand, omega) / (2.0 * np.pi)

        return result

    def verify_parseval(self, x, omega):
        """
        Numerically verify the DTFT form of Parseval's relation:

            sum_n |x[n]|^2
            =
            (1 / 2*pi) integral |X(e^jw)|^2 dw
        """
        x = np.asarray(x, dtype=np.complex128)
        omega = np.asarray(omega, dtype=np.float64)

        X = self.transform(x, omega)

        time_energy = np.sum(np.abs(x) ** 2)

        frequency_energy = (
            np.trapezoid(np.abs(X) ** 2, omega)
            / (2.0 * np.pi)
        )

        error = abs(time_energy - frequency_energy)

        return time_energy, frequency_energy, error

    def is_periodic(self, omega1, omega2, tolerance=1e-10):
        """
        Check whether two angular frequencies are equivalent
        under DTFT periodicity.

        DTFT satisfies:

            X(e^(j(omega + 2*pi))) = X(e^(j*omega))
        """
        difference = np.asarray(omega2) - np.asarray(omega1)

        multiples = difference / (2.0 * np.pi)

        return np.all(
            np.abs(multiples - np.round(multiples)) <= tolerance
        )

    def shift_time(self, x, shift, omega):
        """
        DTFT of a time-shifted signal.

        If

            y[n] = x[n - shift]

        then

            Y(e^jw) = X(e^jw) e^(-j*w*shift)
        """
        X = self.transform(x, omega)

        omega = np.asarray(omega, dtype=np.float64)

        return X * np.exp(-1j * omega * shift)

    def frequency_shift(self, x, omega, omega0):
        """
        DTFT of a frequency-shifted signal.

        If

            y[n] = x[n] e^(j*omega0*n)

        then

            Y(e^jw) = X(e^(j(w-omega0)))
        """
        omega = np.asarray(omega, dtype=np.float64)

        return self.transform(x, omega - omega0)

    def reverse_time(self, x, omega):
        """
        DTFT of a time-reversed signal.

        If

            y[n] = x[-n]

        then

            Y(e^jw) = X(e^(-jw))
        """
        omega = np.asarray(omega, dtype=np.float64)

        return self.transform(x, -omega)

    def conjugate_signal(self, x, omega):
        """
        DTFT of the complex-conjugated signal.

        If

            y[n] = x*[n]

        then

            Y(e^jw) = X*(-jw)
        """
        omega = np.asarray(omega, dtype=np.float64)

        return np.conj(self.transform(x, -omega))

    def frequency_response(self, impulse_response, omega):
        """
        Compute the DTFT of an impulse response h[n].

        For an LTI system:

            H(e^jw) = DTFT{h[n]}

        This is the system frequency response.
        """
        return self.transform(impulse_response, omega)

    def filter_frequency_response(self, x, h, omega):
        """
        Compute output spectrum of an LTI system:

            Y(e^jw) = X(e^jw) H(e^jw)
        """
        X = self.transform(x, omega)
        H = self.transform(h, omega)

        return X * H

    def convolution_property(self, x, h, omega):
        """
        Compute X(e^jw)H(e^jw), corresponding to the DTFT
        of the linear convolution x[n] * h[n].
        """
        X = self.transform(x, omega)
        H = self.transform(h, omega)

        return X * H