"""
transforms.py

Shared transform core.

This file contains:
    - next_power_of_two
    - power-of-two validation
    - zero padding utilities

    Transform engines:
        - DFTAnalyzer
        - FFTTransformer
        - NTTTransformer
        - ArbitraryLengthFFT

    Signal processing utilities:
        - convolution
        - correlation
        - frequency domain filtering
        - magnitude / phase / power spectrum
        - frequency bins
        - fftshift / ifftshift
        - energy verification

    2D operations:
        - 2D DFT
        - 2D FFT
        - 2D convolution

No function in this file calls:
    numpy.fft
    scipy.fft
    numpy.convolve
    scipy.signal

NumPy is used only for array arithmetic.
"""

import numpy as np


NTT_MODULUS = 998244353
NTT_PRIMITIVE_ROOT = 3


# ============================================================
# BASIC UTILITY FUNCTIONS
# ============================================================

def is_power_of_two(n):
    """
    Return True if n is a positive power of two.

    Examples:
        1  -> True
        2  -> True
        8  -> True
        6  -> False
        0  -> False
    """
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= n.

    Examples:
        next_power_of_two(1) -> 1
        next_power_of_two(5) -> 8
        next_power_of_two(8) -> 8
    """
    if n <= 1:
        return 1

    power = 1

    while power < n:
        power *= 2

    return power


def zero_pad(x, length):
    """
    Zero-pad a 1D signal to the requested length.

    Raises ValueError if requested length is smaller.
    """
    x = np.asarray(x)

    if length < len(x):
        raise ValueError("Padding length cannot be smaller than signal length")

    result = np.zeros(length, dtype=x.dtype)
    result[:len(x)] = x

    return result


def zero_pad_to_power_of_two(x):
    """
    Zero-pad a signal to the next power-of-two length.
    """
    x = np.asarray(x)
    N = next_power_of_two(len(x))

    return zero_pad(x, N)


def zero_pad_2d(x, rows, cols):
    """
    Zero-pad a 2D array to shape (rows, cols).
    """
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError("Input must be 2-dimensional")

    if rows < x.shape[0] or cols < x.shape[1]:
        raise ValueError("Padding size cannot be smaller than input size")

    result = np.zeros((rows, cols), dtype=x.dtype)

    result[:x.shape[0], :x.shape[1]] = x

    return result


# ============================================================
# DFT
# ============================================================

class DFTAnalyzer:
    """
    Discrete Fourier Transform computed directly from definition.

        X[k] = sum x[n] exp(-j 2 pi k n / N)

        x[n] = (1/N) sum X[k] exp(+j 2 pi k n / N)
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.
        """
        x = np.asarray(x, dtype=np.complex128)

        N = len(x)

        if N == 0:
            return np.array([], dtype=np.complex128)

        X = np.zeros(N, dtype=np.complex128)

        for k in range(N):
            for n in range(N):
                angle = -2j * np.pi * k * n / N
                X[k] += x[n] * np.exp(angle)

        return X


    def inverse(self, spectrum):
        """
        Inverse DFT.
        """
        spectrum = np.asarray(spectrum, dtype=np.complex128)

        N = len(spectrum)

        if N == 0:
            return np.array([], dtype=np.complex128)

        x = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            for k in range(N):
                angle = 2j * np.pi * k * n / N
                x[n] += spectrum[k] * np.exp(angle)

        return x / N


# ============================================================
# FFT
# ============================================================

class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 Decimation-In-Time FFT.

    Uses:
        1. Bit reversal permutation
        2. Butterfly computation
        3. Stage-wise twiddle factors
    """

    name = "fft"


    def _check_length(self, N):
        """
        Validate FFT length.
        """
        if not is_power_of_two(N):
            raise ValueError("N must be a positive power of two")


    def _bit_reverse(self, values):
        """
        Return bit-reversed permutation of the array.
        """
        values = values.copy()

        N = len(values)

        j = 0

        for i in range(1, N):
            bit = N >> 1

            while j & bit:
                j ^= bit
                bit >>= 1

            j ^= bit

            if i < j:
                values[i], values[j] = values[j], values[i]

        return values


    def transform(self, x):
        """
        Forward radix-2 FFT.
        """
        X = np.asarray(x, dtype=np.complex128).copy()

        N = len(X)

        self._check_length(N)

        # Bit reversal
        X = self._bit_reverse(X)

        # Butterfly stages
        size = 2

        while size <= N:

            half = size // 2

            # Twiddle factor for this stage
            stage_twiddle = np.exp(-2j * np.pi / size)

            for start in range(0, N, size):

                W = 1.0 + 0.0j

                for k in range(half):

                    even = X[start + k]
                    odd = W * X[start + k + half]

                    X[start + k] = even + odd
                    X[start + k + half] = even - odd

                    W *= stage_twiddle

            size *= 2

        return X


    def inverse(self, spectrum):
        """
        Inverse FFT using:

            IFFT(X) = conjugate(FFT(conjugate(X))) / N
        """
        spectrum = np.asarray(spectrum, dtype=np.complex128)

        N = len(spectrum)

        self._check_length(N)

        conjugated = np.conj(spectrum)

        transformed = self.transform(conjugated)

        result = np.conj(transformed) / N

        return result


# ============================================================
# NTT
# ============================================================

class NTTTransformer:
    """
    Exact radix-2 Number Theoretic Transform.

    Useful for exact integer convolution.
    """

    name = "ntt"

    modulus = NTT_MODULUS
    primitive_root = NTT_PRIMITIVE_ROOT


    def _transform(self, values, inverse):

        values = np.asarray(values, dtype=np.int64).copy()

        N = len(values)

        if not is_power_of_two(N):
            raise ValueError("N must be a positive power of two")

        values %= self.modulus

        # Bit reversal
        j = 0

        for i in range(1, N):

            bit = N >> 1

            while j & bit:
                j ^= bit
                bit >>= 1

            j ^= bit

            if i < j:
                values[i], values[j] = values[j], values[i]

        # Butterfly stages
        size = 2

        while size <= N:

            root = pow(
                self.primitive_root,
                (self.modulus - 1) // size,
                self.modulus
            )

            if inverse:
                root = pow(
                    root,
                    self.modulus - 2,
                    self.modulus
                )

            half = size // 2

            for start in range(0, N, size):

                twiddle = 1

                for offset in range(half):

                    even = int(values[start + offset])

                    odd = (
                        twiddle *
                        int(values[start + offset + half])
                    ) % self.modulus

                    values[start + offset] = (
                        even + odd
                    ) % self.modulus

                    values[start + offset + half] = (
                        even - odd
                    ) % self.modulus

                    twiddle = (
                        twiddle * root
                    ) % self.modulus

            size *= 2

        # Scale inverse
        if inverse:

            scale = pow(
                N,
                self.modulus - 2,
                self.modulus
            )

            values = (
                values * scale
            ) % self.modulus

        return values.astype(np.int64)


    def transform(self, x):
        """
        Forward NTT.
        """
        return self._transform(x, inverse=False)


    def inverse(self, spectrum):
        """
        Inverse NTT.
        """
        return self._transform(spectrum, inverse=True)


# ============================================================
# ARBITRARY LENGTH FFT (BLUESTEIN)
# ============================================================

class ArbitraryLengthFFT(FFTTransformer):
    """
    FFT for arbitrary length signals using Bluestein's algorithm.

    Works for:
        N = 3
        N = 5
        N = 6
        N = 7
        ...

    while internally using radix-2 FFT.
    """

    name = "arbitrary"


    def transform(self, x):

        x = np.asarray(x, dtype=np.complex128)

        N = len(x)

        if N == 0:
            return np.array([], dtype=np.complex128)

        if N == 1:
            return x.copy()

        # Already power of two
        if is_power_of_two(N):
            return super().transform(x)

        # Convolution length
        M = next_power_of_two(2 * N - 1)

        n = np.arange(N)

        # Bluestein chirp
        chirp = np.exp(
            -1j * np.pi * n * n / N
        )

        positive_chirp = np.exp(
            1j * np.pi * n * n / N
        )

        a = np.zeros(
            M,
            dtype=np.complex128
        )

        b = np.zeros(
            M,
            dtype=np.complex128
        )

        a[:N] = x * chirp

        b[:N] = positive_chirp

        for i in range(1, N):
            b[M - i] = positive_chirp[i]

        # FFT convolution
        A = super().transform(a)
        B = super().transform(b)

        C = A * B

        c = super().inverse(C)

        X = c[:N] * chirp

        return X


    def inverse(self, spectrum):

        spectrum = np.asarray(
            spectrum,
            dtype=np.complex128
        )

        N = len(spectrum)

        if N == 0:
            return np.array([], dtype=np.complex128)

        conjugated = np.conj(spectrum)

        transformed = self.transform(conjugated)

        result = np.conj(transformed) / N

        return result


# ============================================================
# CONVOLUTION UTILITIES
# ============================================================

class ConvolutionAnalyzer:
    """
    Different methods of convolution.

    Supports:
        - Direct convolution
        - Circular convolution
        - Linear convolution using FFT
    """


    @staticmethod
    def direct(x, h):
        """
        Direct linear convolution.

        y[n] = sum x[k] h[n-k]
        """
        x = np.asarray(x)
        h = np.asarray(h)

        N = len(x)
        M = len(h)

        result = np.zeros(
            N + M - 1,
            dtype=np.complex128
        )

        for n in range(N + M - 1):

            total = 0

            for k in range(N):

                j = n - k

                if 0 <= j < M:
                    total += x[k] * h[j]

            result[n] = total

        return result


    @staticmethod
    def circular(x, h, engine=None):
        """
        Circular convolution using a transform.

        Both signals are padded to equal length.
        """

        if engine is None:
            engine = FFTTransformer()

        x = np.asarray(x, dtype=np.complex128)
        h = np.asarray(h, dtype=np.complex128)

        N = max(len(x), len(h))

        if not is_power_of_two(N):
            N = next_power_of_two(N)

        x_pad = zero_pad(x, N)
        h_pad = zero_pad(h, N)

        X = engine.transform(x_pad)
        H = engine.transform(h_pad)

        Y = X * H

        return engine.inverse(Y)


    @staticmethod
    def linear_fft(x, h, engine=None):
        """
        Linear convolution using FFT.

        Required transform length:

            N >= len(x) + len(h) - 1
        """

        if engine is None:
            engine = FFTTransformer()

        x = np.asarray(x, dtype=np.complex128)
        h = np.asarray(h, dtype=np.complex128)

        output_length = len(x) + len(h) - 1

        if isinstance(engine, ArbitraryLengthFFT):
            N = output_length
        else:
            N = next_power_of_two(output_length)

        x_pad = zero_pad(x, N)
        h_pad = zero_pad(h, N)

        X = engine.transform(x_pad)
        H = engine.transform(h_pad)

        Y = X * H

        result = engine.inverse(Y)

        return result[:output_length]


# ============================================================
# CORRELATION
# ============================================================

class CorrelationAnalyzer:
    """
    Cross-correlation and autocorrelation.
    """


    @staticmethod
    def direct(x, y):
        """
        Direct cross-correlation.

        r_xy[l] = sum x[n] * conj(y[n-l])

        Output length:
            len(x) + len(y) - 1
        """

        x = np.asarray(x, dtype=np.complex128)
        y = np.asarray(y, dtype=np.complex128)

        N = len(x)
        M = len(y)

        result = np.zeros(
            N + M - 1,
            dtype=np.complex128
        )

        for lag in range(-(M - 1), N):

            total = 0

            for n in range(N):

                index = n - lag

                if 0 <= index < M:
                    total += x[n] * np.conj(y[index])

            result[lag + M - 1] = total

        return result


    @staticmethod
    def circular(x, y, engine=None):
        """
        Circular cross-correlation using:

            Rxy = X * conjugate(Y)
        """

        if engine is None:
            engine = FFTTransformer()

        x = np.asarray(x, dtype=np.complex128)
        y = np.asarray(y, dtype=np.complex128)

        N = max(len(x), len(y))

        N = next_power_of_two(N)

        x_pad = zero_pad(x, N)
        y_pad = zero_pad(y, N)

        X = engine.transform(x_pad)
        Y = engine.transform(y_pad)

        R = X * np.conj(Y)

        return engine.inverse(R)


    @staticmethod
    def autocorrelation(x, engine=None):
        """
        Autocorrelation of x.
        """

        return CorrelationAnalyzer.circular(
            x,
            x,
            engine
        )


# ============================================================
# SPECTRUM ANALYSIS
# ============================================================

class SpectrumAnalyzer:
    """
    Helper methods for analyzing a Fourier spectrum.
    """


    @staticmethod
    def magnitude(X):
        """
        Magnitude spectrum:

            |X[k]|
        """
        X = np.asarray(X)
        return np.abs(X)


    @staticmethod
    def phase(X):
        """
        Phase spectrum in radians.
        """
        X = np.asarray(X)
        return np.angle(X)


    @staticmethod
    def power(X):
        """
        Power spectrum:

            |X[k]|^2
        """
        X = np.asarray(X)
        return np.abs(X) ** 2


    @staticmethod
    def log_magnitude(X, epsilon=1e-12):
        """
        Log magnitude spectrum.

        Useful for visualization.
        """

        magnitude = np.abs(X)

        return np.log(
            magnitude + epsilon
        )


    @staticmethod
    def frequency_bins(N, sampling_frequency=1.0):
        """
        Return frequency corresponding to every DFT bin.

        Frequency resolution:

            df = fs / N

        DFT bin frequency:

            f[k] = k * fs / N
        """

        return np.arange(N) * (
            sampling_frequency / N
        )


    @staticmethod
    def centered_frequency_bins(
        N,
        sampling_frequency=1.0
    ):
        """
        Frequency bins after fftshift.

        Produces negative frequencies first.
        """

        frequencies = np.zeros(N)

        df = sampling_frequency / N

        half = N // 2

        for i in range(N):

            if i < half:
                frequencies[i] = (
                    (i - half) * df
                )
            else:
                frequencies[i] = (
                    (i - half) * df
                )

        return frequencies


# ============================================================
# FFT SHIFT
# ============================================================

def fftshift(x):
    """
    Shift zero frequency component to center.

    Example:

        [0,1,2,3,4,5,6,7]

    becomes:

        [4,5,6,7,0,1,2,3]
    """

    x = np.asarray(x)

    N = len(x)

    shift = N // 2

    return np.concatenate((
        x[shift:],
        x[:shift]
    ))


def ifftshift(x):
    """
    Inverse of fftshift.
    """

    x = np.asarray(x)

    N = len(x)

    shift = (N + 1) // 2

    return np.concatenate((
        x[shift:],
        x[:shift]
    ))


def fftshift_2d(x):
    """
    2D FFT shift.

    Moves DC component to image center.
    """

    x = np.asarray(x)

    rows, cols = x.shape

    row_shift = rows // 2
    col_shift = cols // 2

    result = np.concatenate(
        (x[row_shift:, :], x[:row_shift, :]),
        axis=0
    )

    result = np.concatenate(
        (
            result[:, col_shift:],
            result[:, :col_shift]
        ),
        axis=1
    )

    return result


# ============================================================
# FREQUENCY DOMAIN FILTERING
# ============================================================

class FrequencyFilter:
    """
    Frequency domain filtering utilities.
    """


    @staticmethod
    def apply(signal, filter_response, engine=None):
        """
        Apply a frequency-domain filter.

        Steps:
            1. FFT(signal)
            2. Multiply by H[k]
            3. IFFT
        """

        if engine is None:
            engine = FFTTransformer()

        signal = np.asarray(
            signal,
            dtype=np.complex128
        )

        filter_response = np.asarray(
            filter_response,
            dtype=np.complex128
        )

        if len(signal) != len(filter_response):
            raise ValueError(
                "Signal and filter response must have same length"
            )

        X = engine.transform(signal)

        Y = X * filter_response

        return engine.inverse(Y)


    @staticmethod
    def low_pass(N, cutoff_bin):
        """
        Create an ideal low-pass filter.

        Keeps frequencies near DC.
        """

        H = np.zeros(
            N,
            dtype=np.complex128
        )

        H[:cutoff_bin + 1] = 1

        if cutoff_bin > 0:
            H[-cutoff_bin:] = 1

        return H


    @staticmethod
    def high_pass(N, cutoff_bin):
        """
        Create an ideal high-pass filter.
        """

        H = np.ones(
            N,
            dtype=np.complex128
        )

        H[:cutoff_bin + 1] = 0

        if cutoff_bin > 0:
            H[-cutoff_bin:] = 0

        return H


    @staticmethod
    def band_pass(N, low_bin, high_bin):
        """
        Create an ideal band-pass filter.
        """

        H = np.zeros(
            N,
            dtype=np.complex128
        )

        H[low_bin:high_bin + 1] = 1

        if low_bin > 0:
            H[-high_bin:-low_bin + 1] = 1

        return H


# ============================================================
# ENERGY AND PARSEVAL VERIFICATION
# ============================================================

class TransformVerifier:
    """
    Functions for verifying transform properties.
    """


    @staticmethod
    def reconstruction_error(x, engine):
        """
        Measure reconstruction error:

            x -> Transform -> Inverse -> x_hat
        """

        x = np.asarray(
            x,
            dtype=np.complex128
        )

        X = engine.transform(x)

        reconstructed = engine.inverse(X)

        return np.max(
            np.abs(reconstructed - x)
        )


    @staticmethod
    def compare_dft_fft(x):
        """
        Compare direct DFT and FFT.
        """

        x = np.asarray(
            x,
            dtype=np.complex128
        )

        dft = DFTAnalyzer()
        fft = FFTTransformer()

        X_dft = dft.transform(x)
        X_fft = fft.transform(x)

        return np.max(
            np.abs(X_dft - X_fft)
        )


    @staticmethod
    def time_energy(x):
        """
        Energy in time domain:

            sum |x[n]|^2
        """

        x = np.asarray(x)

        return np.sum(
            np.abs(x) ** 2
        )


    @staticmethod
    def frequency_energy(X):
        """
        Energy in frequency domain.

        According to Parseval:

            sum |x[n]|^2
            =
            (1/N) sum |X[k]|^2
        """

        X = np.asarray(X)

        N = len(X)

        return np.sum(
            np.abs(X) ** 2
        ) / N


    @staticmethod
    def parseval_error(x, engine):
        """
        Verify Parseval's theorem.
        """

        X = engine.transform(x)

        time_energy = TransformVerifier.time_energy(x)

        frequency_energy = (
            TransformVerifier.frequency_energy(X)
        )

        return abs(
            time_energy - frequency_energy
        )


# ============================================================
# 2D TRANSFORMS
# ============================================================

class Transform2D:
    """
    2D DFT / FFT.

    A 2D transform is computed by:

        1. Transform every row
        2. Transform every column
    """


    @staticmethod
    def transform(x, engine=None):
        """
        Forward 2D transform.
        """

        if engine is None:
            engine = FFTTransformer()

        x = np.asarray(
            x,
            dtype=np.complex128
        )

        if x.ndim != 2:
            raise ValueError(
                "Input must be a 2D array"
            )

        rows, cols = x.shape

        result = x.copy()

        # Transform rows
        for row in range(rows):
            result[row, :] = engine.transform(
                result[row, :]
            )

        # Transform columns
        for col in range(cols):
            result[:, col] = engine.transform(
                result[:, col]
            )

        return result


    @staticmethod
    def inverse(spectrum, engine=None):
        """
        Inverse 2D transform.
        """

        if engine is None:
            engine = FFTTransformer()

        spectrum = np.asarray(
            spectrum,
            dtype=np.complex128
        )

        rows, cols = spectrum.shape

        result = spectrum.copy()

        # Inverse rows
        for row in range(rows):
            result[row, :] = engine.inverse(
                result[row, :]
            )

        # Inverse columns
        for col in range(cols):
            result[:, col] = engine.inverse(
                result[:, col]
            )

        return result


# ============================================================
# 2D CONVOLUTION
# ============================================================

class Convolution2D:
    """
    2D convolution using Fourier transforms.
    """


    @staticmethod
    def direct(image, kernel):
        """
        Direct full 2D linear convolution.
        """

        image = np.asarray(image)
        kernel = np.asarray(kernel)

        image_rows, image_cols = image.shape
        kernel_rows, kernel_cols = kernel.shape

        output_rows = (
            image_rows + kernel_rows - 1
        )

        output_cols = (
            image_cols + kernel_cols - 1
        )

        output = np.zeros(
            (output_rows, output_cols),
            dtype=np.complex128
        )

        for i in range(image_rows):

            for j in range(image_cols):

                for ki in range(kernel_rows):

                    for kj in range(kernel_cols):

                        output[i + ki, j + kj] += (
                            image[i, j] *
                            kernel[ki, kj]
                        )

        return output


    @staticmethod
    def fft(image, kernel, engine=None):
        """
        2D linear convolution using FFT.
        """

        if engine is None:
            engine = FFTTransformer()

        image = np.asarray(
            image,
            dtype=np.complex128
        )

        kernel = np.asarray(
            kernel,
            dtype=np.complex128
        )

        rows = (
            image.shape[0] +
            kernel.shape[0] - 1
        )

        cols = (
            image.shape[1] +
            kernel.shape[1] - 1
        )

        if isinstance(engine, ArbitraryLengthFFT):
            transform_rows = rows
            transform_cols = cols
        else:
            transform_rows = next_power_of_two(rows)
            transform_cols = next_power_of_two(cols)

        image_pad = zero_pad_2d(
            image,
            transform_rows,
            transform_cols
        )

        kernel_pad = zero_pad_2d(
            kernel,
            transform_rows,
            transform_cols
        )

        image_spectrum = Transform2D.transform(
            image_pad,
            engine
        )

        kernel_spectrum = Transform2D.transform(
            kernel_pad,
            engine
        )

        result_spectrum = (
            image_spectrum *
            kernel_spectrum
        )

        result = Transform2D.inverse(
            result_spectrum,
            engine
        )

        return result[:rows, :cols]


# ============================================================
# SIMPLE SELF TEST
# ============================================================

def run_self_test():
    """
    Run basic transform verification tests.
    """

    x = (
        np.random.randn(64)
        +
        1j * np.random.randn(64)
    )

    dft = DFTAnalyzer()
    fft = FFTTransformer()

    X_dft = dft.transform(x)
    X_fft = fft.transform(x)

    difference = np.max(
        np.abs(X_dft - X_fft)
    )

    reconstruction = np.max(
        np.abs(
            dft.inverse(X_dft) - x
        )
    )

    print("DFT vs FFT error:", difference)

    print(
        "DFT reconstruction error:",
        reconstruction
    )

    assert difference < 1e-9
    assert reconstruction < 1e-9

    print("All tests passed.")


if __name__ == "__main__":
    run_self_test()