import math
import cmath
import numpy as np
from transforms import FFTTransformer, next_power_of_two
def fft(a):
    fft=    FFTTransformer()
    return fft.transform(a)

def ifft(a):
    fft = FFTTransformer()
    return fft.inverse(a)

def weighted_polynomial_multiply(P,Q,W):
    P=P[::-1]
    Q=Q[::-1]
    W=W[::-1]
    m=len(P)
    n=len(Q)
    for i in range(m):
        P[i] = P[i]*W[i]
    N=next_power_of_two(m+n-1)
    P_padded = np.zeros(N,dtype = np.complex128)
    Q_padded = np.zeros(N,dtype = np.complex128)
    P_padded[:m]=P
    Q_padded[:n] = Q
    R=ifft(fft(P_padded)*fft(Q_padded)).real
    return R[:m+n-1]
# def weighted_polynomial_multiply(P, Q, W):
#     # implement
#     m = len(P)
#     n = len(Q)

#     # Reverse because inputs are given in descending powers.
#     P = P[::-1]
#     Q = Q[::-1]
#     W = W[::-1]

#     # Apply weights to P.
#     P_weighted = np.zeros(m, dtype=np.float64)

#     for i in range(m):
#         P_weighted[i] = P[i] * W[i]

#     # Need at least m+n-1 points for linear convolution.
#     N = next_power_of_two(m + n - 1)

#     P_padded = np.zeros(N, dtype=np.complex128)
#     Q_padded = np.zeros(N, dtype=np.complex128)

#     P_padded[:m] = P_weighted
#     Q_padded[:n] = Q

#     # FFT
#     r = fft(P_padded) * fft(Q_padded)

#     # IFFT
#     R = ifft(r)

#     # Keep only the linear convolution coefficients.
#     R = R[:m+n-1]

#     return R.real

if __name__ == "__main__":
    P = [1, 3, 2, 6, 7]
    Q = [4,1]
    W = [3, 2, 1, 5, 6]
 

    R = weighted_polynomial_multiply(P, Q, W)

    print("Result:", R)