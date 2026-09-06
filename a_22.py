"""
Big Integer Multiplication using FFT / IFFT
--------------------------------------------

You are given two large non-negative integers as arrays of
base-10 digits in LSD-first order.

Example:
    123 -> [3, 2, 1]
     45 -> [5, 4]

Your task is to compute their product using:

    digit arrays
        ↓
    zero padding
        ↓
    FFT
        ↓
    pointwise multiplication
        ↓
    IFFT
        ↓
    convolution coefficients
        ↓
    carry propagation
        ↓
    LSD-first result

DO NOT use:
    - numpy.fft
    - numpy.convolve
    - Python's big-integer multiplication

You MAY use the FFT implementation from transforms.py.
"""

import numpy as np

from transforms import FFTTransformer, next_power_of_two


# ============================================================
# 1. INPUT / OUTPUT HELPERS
# ============================================================

def number_to_digits(number):
    """
    Convert a non-negative integer/string into an LSD-first
    array of decimal digits.

    Example:
        "123" -> [3, 2, 1]
        "45"  -> [5, 4]

    Parameters
    ----------
    number : str or int

    Returns
    -------
    numpy.ndarray
        LSD-first decimal digits.
    """

    # TODO 1:
    # Convert the input into a string.
    #
    number  = str(number)
    # TODO 2:
    # Convert each character into a decimal digit.
    #
    nums = []
    for d in number:
        nums.append(int(d))
    nums = np.asarray(nums,dtype=np.int64)
    nums = nums[::-1]
    return nums
    # TODO 3:
    # Reverse the order so that the result is LSD-first.
    #
    # Example:
    # "123" -> ['1','2','3']
    #       -> [1,2,3]
    #       -> [3,2,1]



def digits_to_number(digits):
    """
    Convert an LSD-first digit array back into a normal
    decimal string.

    Example:
        [5, 3, 5, 5] -> "5535"

    Parameters
    ----------
    digits : array-like

    Returns
    -------
    str
        Normal decimal representation.
    """

    # TODO 4:
    # Convert every digit to a string.
    nums = []
    for d in digits:
        nums.append(str(d))
    # TODO 5:
    # Reverse the digit order.
    nums = nums[::-1]
    # TODO 6:
    # Join the digits together.

    number =""
    for n in nums:
        number=number+n
    return number


# ============================================================
# 2. ZERO PADDING
# ============================================================

def pad_for_convolution(a, b):
    """
    Zero-pad two digit arrays so that their circular
    convolution is equivalent to their linear convolution.

    If:
        len(a) = m
        len(b) = n

    then the linear convolution has:

        m + n - 1

    coefficients.

    The FFT length N must therefore satisfy:

        N >= m + n - 1

    For radix-2 FFT, choose the next power of two.
    """

    # TODO 7:
    # Calculate the minimum required convolution length.
    N = len(a)+len(b)-1
    N = next_power_of_two(N)
    # TODO 8:
    # Find the next power of two.

    # TODO 9:
    # Create two zero-filled arrays of length N.
    a_padded = np.zeros(N,dtype=np.float64)
    b_padded = np.zeros(N,dtype=np.float64)
    # TODO 10:
    # Copy a and b into the beginning of the padded arrays.
    a_padded[:len(a)]=a
    b_padded[:len(b)]=b
    # TODO 11:
    # Return:
    #
    #     a_padded, b_padded, N

    return a_padded, b_padded, N


# ============================================================
# 3. FFT-BASED CIRCULAR CONVOLUTION
# ============================================================

def circular_convolution_fft(a, b):
    """
    Compute the convolution of a and b using:

        FFT -> pointwise multiplication -> IFFT

    The inputs should already be padded sufficiently so that
    circular convolution does not cause aliasing.

    Returns
    -------
    numpy.ndarray
        Raw convolution coefficients.
    """

    # TODO 12:
    # Create an FFTTransformer object.
    fft = FFTTransformer()
    # TODO 13:
    # Compute:
    #
    #     A = FFT(a)
    #     B = FFT(b)
    A = fft.transform(a)
    B = fft.transform(b)
    # TODO 14:
    # Multiply the spectra pointwise:
    #
    #     C = A * B
    C = A*B
    # TODO 15:
    # Apply the inverse FFT:
    #
    #     c = IFFT(C)
    c = fft.inverse(C).real
    # TODO 16:
    # The result should theoretically be real-valued.
    #
    # Due to floating-point errors, take the real part.

    # TODO 17:
    # Round the result to the nearest integer.
    #
    # Why?
    # FFT/IFFT may produce values such as:
    #
    #     14.999999999
    #     22.000000001
    #
    # instead of exact integers.

    # TODO 18:
    # Return the raw convolution coefficients.

    return np.round(c)


# ============================================================
# 4. CARRY PROPAGATION
# ============================================================

def propagate_carries(coefficients):
    """
    Convert raw polynomial coefficients into valid decimal
    digits.

    Example:

        coefficients:
        [15, 22, 13, 4]

    becomes:

        [5, 3, 5, 5]

    because:

        index 0:
            15 -> digit 5, carry 1

        index 1:
            22 + 1 -> 23
            digit 3, carry 2

        ...

    Parameters
    ----------
    coefficients : array-like
        Raw convolution coefficients.

    Returns
    -------
    numpy.ndarray
        LSD-first decimal digits.
    """

    # TODO 19:
    # Make a copy of the coefficients so that you do not
    # modify the original array.
    coeffs = np.array(coefficients,dtype = np.int64).copy()
    # TODO 20:
    # Start with:
    #
    #     carry = 0
    carry = 0
    N = len(coeffs)
    # TODO 21:
    # Iterate from index 0 to the end.
    for i in range(0,N):
        value = coeffs[i] + carry
        digit =  value%10
        carry = value//10
        coeffs[i] = digit
    # TODO 22:
    # At every index:
    #
    #     value = coefficient + carry
    #
    # Then determine:
    #
    #     digit = value % 10
    #     carry = value // 10
    while(carry>0):
        coeffs = np.append(coeffs,carry%10)
        carry = carry //10
        
    # TODO 23:
    # Store the digit back at the current index.

    # TODO 24:
    # After the loop, check whether carry is still non-zero.
    #
    # If it is, append its decimal digits appropriately.

    # TODO 25:
    # Remove unnecessary zeros from the MOST significant end.
    #
    # But make sure that zero itself is represented as:
    #
    #     [0]

    # TODO 26:
    # Return the normalized LSD-first digit array.

    while len(coeffs)>1 and coeffs[-1]==0:
        coeffs = coeffs[:-1]
    return coeffs


# ============================================================
# 5. COMPLETE MULTIPLICATION
# ============================================================

def multiply_big_integers(A, B):
    """
    Multiply two large non-negative integers represented as
    LSD-first decimal digit arrays.

    Example:

        A = [3, 2, 1]
        B = [5, 4]

        result = [5, 3, 5, 5]
    """

    # TODO 27:
    # Determine the lengths:
    #
    #     m = len(A)
    #     n = len(B)
    m = len(A)
    n= len(B)
    # Pad A and B sufficiently for linear convolution.
    A,B,N=pad_for_convolution(A,B)
    C=circular_convolution_fft(A,B)
    # TODO 29:
    # Compute the FFT-based circular convolution.
    C=C[:(m+n-1)]
    # TODO 30:
    # Keep only the first:
    #
    #     m + n - 1
    #
    # coefficients.
    #
    # Why is this necessary?

    # TODO 31:
    # Propagate decimal carries.
    nums = propagate_carries(C)
    # TODO 32:
    # Return the final LSD-first digit array.

    return nums


# ============================================================
# 6. TESTING
# ============================================================

def print_multiplication(A, B):
    """
    Helper for testing your implementation.
    """

    result = multiply_big_integers(A, B)

    print("A      :", A)
    print("B      :", B)
    print("Result :", result)
    print()


def main():

    # --------------------------------------------------------
    # Example 1
    # --------------------------------------------------------

    A = np.array([3, 2, 1])
    B = np.array([5, 4])

    print("Example 1")
    print_multiplication(A, B)

    # Expected:
    #
    # Result : [5, 3, 5, 5]
    #
    # DO NOT hard-code this result.


    # --------------------------------------------------------
    # Example 2
    # --------------------------------------------------------

    A = np.array([9, 9, 9])
    B = np.array([9, 9])

    print("Example 2")
    print_multiplication(A, B)

    # Expected:
    #
    # Result : [1, 0, 9, 8, 9]


    # --------------------------------------------------------
    # Additional tests
    # --------------------------------------------------------

    # TODO 33:
    # Test:
    #
    #     1 * 1
    #
    # represented as:
    #
    #     [1] * [1]


    # TODO 34:
    # Test:
    #
    #     9 * 9


    # TODO 35:
    # Test a multiplication where there is a carry chain.
    #
    # For example, think about a case where several consecutive
    # coefficients are >= 10.


    # TODO 36:
    # Test multiplication by zero.
    #
    # Example:
    #
    #     [0] * [5, 4, 3]


    # TODO 37:
    # Test numbers with different lengths.


if __name__ == "__main__":
    main()