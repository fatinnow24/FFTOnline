import numpy as np

from transforms import DFTAnalyzer


def circular_convolution(x, h, engine):
    """
    Compute circular convolution using the DFT convolution theorem.

    x and h must have the same length.
    """

    # TODO 1:
    # Convert both inputs to floating-point arrays.
    x = np.asarray(x,dtype=np.float64)
    h = np.asarray(h,dtype = np.float64)
    # TODO 2:
    # Check that both signals are 1D and have the same length.
    if x.ndim!=1 or h.ndim!=1 or len(x)!=len(h):
        raise ValueError("not doable")
    # TODO 3:
    # Compute the DFT of x and h.
    x_dft = engine.transform(x)
    h_dft = engine.transform(h)
    # TODO 4:
    # Multiply their spectra element-by-element.
    spectra = x_dft * h_dft
    # TODO 5:
    # Apply the inverse DFT and return the real component.
    result = engine.inverse(spectra).real
    return result

def main():

    engine = DFTAnalyzer()

    x = np.array([1, 2, 3, 4], dtype=np.float64)
    h = np.array([1, 1, 0, 0], dtype=np.float64)

    result = circular_convolution(x, h, engine)

    print("x:", x)
    print("h:", h)
    print("Circular convolution:", np.round(result, 6))

    # Expected output:
    # x: [1. 2. 3. 4.]
    # h: [1. 1. 0. 0.]
    # Circular convolution: [5. 3. 5. 7.]


if __name__ == "__main__":
    main()
# You are given two 1D real-valued signals \(x[n]\) and \(h[n]\) of the same length \(N\).

# Using the DFT convolution theorem,

# $$ y[n] = \operatorname{IDFT}\{X[k]H[k]\} $$

# where

# $$ X[k]=DFT\{x[n]\},\qquad H[k]=DFT\{h[n]\}. $$

# Complete the function so that it computes the circular convolution of the two signals.

# No numpy.fft or convolution routines may be used.