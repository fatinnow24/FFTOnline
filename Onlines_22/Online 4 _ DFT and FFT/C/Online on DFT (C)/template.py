import cv2
import numpy as np
import math

def fft(x):
    """
    Compute 1D FFT
    """
    pass

def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function
    """
    pass

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
    
    reconstructed_img = None #implement

    print("Reconstructing image using manual FFT...")

    #implement the rest

    

    cv2.imwrite(output_path, reconstructed_img)
    
if __name__ == "__main__":
    reconstruct_image_using_fft("original_image.png", "shifted_image.jpg", "reconstructed_image_fft.jpg")