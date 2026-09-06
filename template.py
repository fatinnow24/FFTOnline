import cv2
import numpy as np
import math
from transforms import FFTTransformer,DFTAnalyzer, next_power_of_two
def fft(x):
    """
    Compute 1D FFT
    """
    # x = np.asarray(x, dtype=complex)
    # N = len(x)

    # if N == 1:
    #     return x

    # if N % 2 != 0:
    #     raise ValueError("FFT input size must be a power of 2")

    # even = fft(x[0::2])
    # odd = fft(x[1::2])

    # result = np.zeros(N, dtype=complex)

    # for k in range(N // 2):
    #     W = np.exp(-2j * np.pi * k / N)

    #     result[k] = even[k] + W * odd[k]
    #     result[k + N // 2] = even[k] - W * odd[k]

    # return result
    dft = DFTAnalyzer()
    return dft.transform(x)


def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function
    """
    # X = np.asarray(X, dtype=complex)
    # N = len(X)

    # # IFFT(X) = conjugate(FFT(conjugate(X))) / N
    # result = np.conj(fft(np.conj(X))) / N

    # return result
    dft = DFTAnalyzer()
    return dft.inverse(X)


def reconstruct_image_using_fft(original_path, shifted_path, output_path):
    
    original_img = cv2.imread(original_path)
    shifted_img = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return
    
    # Convert the original and shifted color images to grayscale.
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)
    
    reconstructed_img = np.zeros_like(orig_gray)

    print("Reconstructing image using manual FFT...")

    rows, cols = orig_gray.shape

    # Process every row separately
    for i in range(rows):

        # Original row = reference signal
        original_row = orig_gray[i, :].astype(float)

        # Shifted row = signal whose shift we want to find
        shifted_row = shift_gray[i, :].astype(float)

        # --------------------------------------------------
        # 1. Compute FFT of both rows
        # --------------------------------------------------

        original_fft = fft(original_row)
        shifted_fft = fft(shifted_row)

        # --------------------------------------------------
        # 2. Compute cross-correlation in frequency domain
        #
        # R = IFFT(conj(X) * Y)
        # --------------------------------------------------

        correlation_frequency = (
            np.conj(original_fft) * shifted_fft
        )

        correlation = ifft(correlation_frequency)

        # --------------------------------------------------
        # 3. Find the location of maximum correlation
        # --------------------------------------------------

        shift = np.argmax(np.real(correlation))

        # --------------------------------------------------
        # 4. Reverse the detected shift
        # --------------------------------------------------

        reconstructed_img[i, :] = np.roll(
            shifted_row,
            -shift
        )

    # Round values and convert to uint8
    reconstructed_img = np.round(reconstructed_img)
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    reconstructed_img = reconstructed_img.astype(np.uint8)

    # Save reconstructed image
    cv2.imwrite(output_path, reconstructed_img)

    print("Reconstruction complete.")
    print("Saved to:", output_path)


if __name__ == "__main__":
    reconstruct_image_using_fft(
        "original_image.png",
        "shifted_image.jpg",
        "reconstructed_image_fft.jpg"
    )