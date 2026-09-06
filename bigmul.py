"""
bigmul.py  --  TASK A: multiplying huge integers with your own transform.

YOUR CODE GOES HERE. The provided modules io_utils.py (file reading, random
operands) and bench_utils.py (timing, runtime plots) are already imported;
everything mathematical is yours.

Usage (the command line is already wired up for you):

    python bigmul.py inputs/1.txt --engine dft --out-dir outputs/1
    python bigmul.py inputs/2.txt --engine dft --out-dir outputs/2
    python bigmul.py inputs/3.txt --engine fft --out-dir outputs/3
    python bigmul.py inputs/4.txt --engine fft --out-dir outputs/4
    python bigmul.py --benchmark --out-dir outputs/benchmark
    
python bigmul.py inputs/1.txt --engine dft --out-dir outputs/task_a/1
python bigmul.py inputs/2.txt --engine dft --out-dir outputs/task_a/2
python bigmul.py inputs/3.txt --engine fft --out-dir outputs/task_a/3
python bigmul.py inputs/4.txt --engine fft --out-dir outputs/task_a/4
python bigmul.py --benchmark --out-dir outputs/task_a/benchmark

Restrictions: no numpy.fft / scipy.fft / numpy.convolve / scipy.signal, and
no Python big-integer multiplication of the operands themselves. Python's
own integers may be used ONLY to check your answer at the end.
"""

import argparse
import os
import sys

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from io_utils import random_decimal, read_operands, write_report, write_text
from transforms import (DFTAnalyzer, FFTTransformer, NTTTransformer,
                        next_power_of_two)

# Python 3.11+ refuses to print integers longer than 4300 digits unless this
# limit is raised, and the verification step below prints one.
sys.set_int_max_str_digits(2_000_000)

# Number of decimal digits packed into one "limb" (one polynomial
# coefficient). The specification explains why 4 is a safe choice and what
# breaks if you raise it too far.
BASE_DIGITS = 4
BASE = 10 ** BASE_DIGITS
NTT_BASE_DIGITS = 1
NTT_BASE = 10 ** NTT_BASE_DIGITS


def to_limbs(text, base_digits=BASE_DIGITS):
    """
    Convert a decimal string into polynomial coefficients.

    "123456789" with base_digits = 4 becomes the little-endian limb array
    [6789, 2345, 1] -- that is, 1*BASE^2 + 2345*BASE^1 + 6789*BASE^0.

    Parameters
    ----------
    text : str
        A decimal integer, possibly with a leading '+' or '-'.
    base_digits : int
        Decimal digits per limb.

    Returns
    -------
    (int, numpy.ndarray)
        The sign (+1 or -1) and the little-endian limb array (dtype int64).
        Handle the sign separately from the magnitude: the transform never
        sees it.
    """
    # TODO: implement this function
    text = text.strip() #remove leading and lagging zeros
    sign = 1 #assume numbers positive
    if (text[0]=='-'): #if negative, overturn sign
        sign = -1
        text = text[1:] #then install the number
    elif (text[0]=='+'): #if positive, sign doesnt require changing
        text = text[1:]
    text = text.lstrip('0') #remove leading 0s cz they are unnecessary
    if (text == ''): #if the number was indeed 0, return sign, 0
        return 1, np.array([0],dtype=np.int64)
    limbs = [] #if not 0, we have ourselves a number
    for end in range(len(text), 0, -base_digits): #for 123456789, this will be (9,0,-4), so 9, 5,1 
        start = max(0, end - base_digits) #first iteration: start = 5
        limbs.append(int(text[start:end])) #first iteration: text[5:9], which gives 6789, appended, next 2345, last 1
    return sign, np.array(limbs, dtype=np.int64) #returns 1, [6789,2345,1] 
    #6789*10000^0 + 2345*10000^1 +1*10000^2 we get-> 6789 + 2345x +x^2, so we get a polynomial behavior

def from_limbs(sign, limbs, base_digits=BASE_DIGITS):
    """
    Convert limbs back into a decimal string, propagating carries.

    The limbs handed to this function are the convolution result, so they are
    NOT yet reduced: an entry may be far larger than BASE. Sweep from the
    least significant limb upwards, carrying the overflow into the next one,
    then strip leading zeros and re-attach the sign.

    Returns
    -------
    str
        The decimal representation. "0" must come out as "0", not "-0" or "".
    """
    # TODO: implement this function
    # after conv, limbs may not be within base range, like [5000,15000,2]
    #this needs to become [5000,5000,3]
    base = 10 ** base_digits #10^4
    limbs = np.asarray(limbs, dtype=np.int64).copy()
    carry = 0 #initially consider carry = 0
    for i in range(len(limbs)):
        value = int(limbs[i]) + carry
        limbs[i] = value % base
        carry = value // base
    while (carry>0):
        limbs = np.append(limbs, carry % base)
        carry //= base
    while(len(limbs) > 1 and limbs[-1] == 0):
        limbs = limbs[:-1]
    if (len(limbs) == 1 and limbs[0] == 0):
        return "0"
    result = str(limbs[-1])
    
    for i in range(len(limbs) -2, -1, -1):
        result += str(limbs[i]).zfill(base_digits)#if 4 digits are not present in each limb now
        #it fills the remaining needed space with 0
    if sign < 0:
        result = "-" + result
    return result

def multiply_transform(a, b, engine):
    """
    Multiply two limb arrays through the frequency domain.

    The product of two polynomials is the LINEAR convolution of their
    coefficients, and the transform gives you convolution as a pointwise
    product -- but a DFT of length N gives you CIRCULAR convolution of period
    N. Choose N large enough that the linear result fits, or the high-order
    coefficients wrap around and silently corrupt the answer.

    Steps:
      1. Choose the transform length N (see above; for FFTTransformer it must
         also be a power of two -- next_power_of_two is in transforms.py).
      2. Zero-pad both limb arrays to length N.
      3. Transform both, multiply the two spectra pointwise, inverse-transform.
      4. The result is real up to rounding error: take the real part and round
         to the nearest integer.

    Parameters
    ----------
    a, b : numpy.ndarray of int64
        Little-endian limb arrays.
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    (numpy.ndarray of int64, int)
        The un-carried convolution coefficients, and the transform length N
        you used (report.txt has to state it).
    """
    # TODO: implement this function
    N = len(a) + len(b) - 1
    N = next_power_of_two(N)
    #this is basically y= x*h but we obtain it using
    #y = Inverse(transform(x)*transform(h))
    #in order to transform, we need complex array as result may be complex
    #however the convolved output may be real
    a_padded = np.zeros(N, dtype=np.complex128)
    b_padded = np.zeros(N, dtype=np.complex128)
    a_padded[:len(a)] = a
    b_padded[:len(b)] = b
    A = engine.transform(a_padded)
    B = engine.transform(b_padded)
    C = A * B
    c = engine.inverse(C)

    result = np.round(np.real(c)).astype(np.int64)
    
    return result, N


def multiply_ntt(a, b):
    N = next_power_of_two(len(a) + len(b) - 1)
    if N >= NTTTransformer.modulus:
        raise ValueError("NTT length is too large for the selected modulus")

    engine = NTTTransformer()
    a_padded = np.zeros(N, dtype=np.int64)
    b_padded = np.zeros(N, dtype=np.int64)
    a_padded[:len(a)] = a
    b_padded[:len(b)] = b
    c=engine.transform(a_padded)*engine.transform(b_padded)
    C=engine.inverse(c)
    return C, N


def multiply_schoolbook(a, b):
    """
    OPTIONAL baseline: the O(n^2) method everyone learns at school, on limbs.

    Only needed if you want a third curve on your runtime plot. Expect it to
    be competitive for a long time: a NumPy-assisted schoolbook multiply has a
    very small constant factor, and constant factors decide who wins at small
    sizes.
    """
    # TODO (optional): implement this function
    N = len(a)+len(b)-1
    result = np.zeros(N, dtype=np.int64)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]

    return result


def multiply(text_a, text_b, method):
    """
    Multiply two decimal strings and return (product_string, N, limbs_a, limbs_b).

    ``method`` is one of "dft", "fft", "ntt", "schoolbook" (optional) or "arbitrary"
    (bonus). Pick the engine, convert to limbs, convolve, carry, re-sign.
    """
    # TODO: implement this function
    if method == "ntt":
        base_digits = NTT_BASE_DIGITS
    else:
        base_digits = BASE_DIGITS
    a_sign, a_limbs = to_limbs(text_a, base_digits)
    b_sign, b_limbs = to_limbs(text_b, base_digits)
    result_sign = a_sign * b_sign
    if method == "dft":
        engine = DFTAnalyzer()
        convolution, N = multiply_transform(a_limbs, b_limbs, engine)
    elif method == "fft":
        engine = FFTTransformer()
        convolution, N = multiply_transform(a_limbs, b_limbs, engine)
    elif method == "arbitrary":
        from transforms import ArbitraryLengthFFT
        engine = ArbitraryLengthFFT()
        convolution, N = multiply_transform(a_limbs, b_limbs, engine)
    elif method == "ntt":
        convolution, N = multiply_ntt(a_limbs, b_limbs)
    elif method == "schoolbook":
        convolution = multiply_schoolbook(a_limbs, b_limbs)
        N = len(convolution)
    else:
        raise ValueError("Unknown multiplication method")

    product = from_limbs(result_sign, convolution, base_digits)
    return product, N, a_limbs, b_limbs


def run_single(path, method, out_dir):
    """
    Process one input file and write the required outputs.

    Must produce, inside ``out_dir``:
      product.txt -- the product as a single decimal string
      report.txt  -- input path, method, digit counts of both operands, the
                     base used, limb counts, the transform length N, the digit
                     count of the product, and the verification verdict.
                     It is written by your code; there is no separate
                     write-up to hand in.

    Verification: compare your product against ``int(text_a) * int(text_b)``.
    This is the ONLY place Python's big integers may be used. Print MATCH or
    MISMATCH; a MISMATCH must not be silently swallowed.
    """
    # TODO: implement this function
    text_a, text_b = read_operands(path)
    product_string, N, a_limbs, b_limbs = multiply(text_a, text_b, method)
    expected = str(int(text_a) * int(text_b))
    if product_string == expected:
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"
    
    print(verdict)
    
    if verdict == "MISMATCH":
        raise ValueError(f"MISMATCH: got {product_string}, expected {expected}")
    
    product_path = os.path.join(out_dir, "product.txt")
    write_text(product_path, product_string)
    
    # Build report in expected format
    digits_a = len(text_a.lstrip('+-'))
    digits_b = len(text_b.lstrip('+-'))
    limbs_count_a = len(a_limbs)
    limbs_count_b = len(b_limbs)
    product_digits = len(product_string.lstrip('+-'))
    if method == "ntt":
        base_digits = NTT_BASE_DIGITS
    else:
        base_digits = BASE_DIGITS
    report_lines = [
        "Task A -- big-integer multiplication by spectral convolution",
        f"input file          : {path}",
        f"method              : {method}",
        f"digits of A / B     : {digits_a} / {digits_b}",
        f"base                : 10^{base_digits}",
        f"limbs of A / B      : {limbs_count_a} / {limbs_count_b}",
        f"transform length N  : {N}",
        f"digits of product   : {product_digits}",
        f"verification        : {verdict}",
    ]
    report_path = os.path.join(out_dir, "report.txt")
    write_report(report_path, report_lines)



# ---------------------------------------------------------------------------
# PROVIDED -- run_benchmark is already written. It calls your multiply(), so
# it starts working as soon as your transforms and multiply() are correct.
# You do not need to modify anything below (though you may extend it).
# ---------------------------------------------------------------------------
DFT_SIZES = [128, 256, 512, 1024, 2048, 4096]
FFT_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
TIME_BUDGET = 20.0          # stop a sweep once one measurement exceeds this


def run_benchmark(out_dir):
    """
    Measure naive DFT against radix-2 FFT over growing operands and write
    runtime_bigmul.png plus the timing table in report.txt.

    Each sweep stops early once a single measurement exceeds TIME_BUDGET
    seconds, so a slow machine (or a slow DFT) simply produces a shorter
    curve rather than hanging.
    """
    def measure(label, method, sizes):
        xs, ys = [], []
        print("%s:" % label)
        for digits in sizes:
            a = random_decimal(digits, seed=digits)
            b = random_decimal(digits, seed=digits + 1)
            seconds = time_best(lambda: multiply(a, b, method), repeats=2)
            xs.append(digits)
            ys.append(seconds)
            print("  %8d digits   %9.4f s" % (digits, seconds))
            if seconds > TIME_BUDGET:
                print("  (stopping this curve -- over the time budget)")
                break
        return xs, ys

    series = {}
    series["Naive DFT"] = measure("naive DFT", "dft", DFT_SIZES)
    series["Radix-2 FFT"] = measure("radix-2 FFT", "fft", FFT_SIZES)
    try:                                    # optional third curve
        series["Schoolbook"] = measure("schoolbook", "schoolbook", FFT_SIZES)
    except NotImplementedError:
        series.pop("Schoolbook", None)
        print("schoolbook: not implemented, skipping that curve")

    plot_path = os.path.join(out_dir, "runtime_bigmul.png")
    plot_runtime_curve(series, plot_path,
                       title="Task A: big-integer multiplication",
                       xlabel="decimal digits per operand",
                       references=("n2", "nlogn"))
    write_report(os.path.join(out_dir, "report.txt"),
                 ["Task A -- runtime benchmark", ""]
                 + timing_table_lines(series, size_label="digits")
                 + ["", "plot: %s" % os.path.basename(plot_path)])
    print("wrote", plot_path)


def main():
    ap = argparse.ArgumentParser(description="Big-integer multiplication by DFT/FFT")
    ap.add_argument("input", nargs="?", help="input file with the two operands")
    ap.add_argument("--engine", default="fft",
                    choices=["dft", "fft", "ntt", "schoolbook", "arbitrary"])
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the timing study instead of a single multiplication")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.benchmark:
        run_benchmark(args.out_dir)
    else:
        if not args.input:
            ap.error("an input file is required unless --benchmark is given")
        run_single(args.input, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
