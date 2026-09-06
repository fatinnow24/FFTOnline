"""
image_conv.py  --  TASK B: blurring an image through the frequency domain.

YOUR CODE GOES HERE. image_utils.py (loading, saving, kernels, comparison
figures) and bench_utils.py (timing, runtime plots) are provided; the
transform, the padding logic and the convolution are yours.

Usage (the command line is already wired up for you):

    python image_conv.py images/skyline512.png --kernel bokeh --param 9 --engine fft --out-dir outputs/skyline_bokeh
    python image_conv.py images/sunset512.png --gray --kernel motion --param 41 --engine fft --out-dir outputs/sunset_motion
    python image_conv.py images/nebula512.png --kernel bokeh --param 13 --engine fft --out-dir outputs/nebula_bokeh
    python image_conv.py images/skyline256.png --gray --kernel gaussian --param 21 --engine dft --out-dir outputs/skyline256_gaussian_dft
    python image_conv.py images/skyline512.png --benchmark --out-dir outputs/benchmark

    python image_conv.py images/skyline512.png --kernel bokeh --param 9 --engine fft --out-dir outputs/task_b/skyline_bokeh
    python image_conv.py images/sunset512.png --gray --kernel motion --param 41 --engine fft --out-dir outputs/task_b/sunset_motion
    python image_conv.py images/nebula512.png --kernel bokeh --param 13 --engine fft --out-dir outputs/task_b/nebula_bokeh
    python image_conv.py images/skyline256.png --gray --kernel gaussian --param 21 --engine dft --out-dir outputs/task_b/skyline256_gaussian_dft
    python image_conv.py images/skyline512.png --benchmark --out-dir outputs/task_b/benchmark
    
Restrictions: no numpy.fft / scipy.fft / scipy.ndimage / cv2 / PIL filters,
no numpy.convolve, no scipy.signal. Every transform goes through your own
transforms.py.
"""

import argparse
import os

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from image_utils import (load_image, make_kernel, save_comparison, save_image,
                         save_kernel_preview)
from io_utils import write_report
from transforms import DFTAnalyzer, FFTTransformer, ArbitraryLengthFFT, next_power_of_two


def transform_2d(plane, engine):
    """
    2D forward transform of a single 2D array, by separability.

    The 2D DFT factorises into 1D transforms: transform every ROW, then
    transform every COLUMN of the result (the order does not matter). This is
    the only reason a 2D transform is affordable -- doing it directly from the
    2D definition would be O(N^4).

    Parameters
    ----------
    plane : 2D array_like, shape (P, Q)
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of complex128, shape (P, Q)
    """
    # TODO: implement this function
    plane = np.asarray(plane, dtype = np.complex128)
    rows,cols = plane.shape
    result = np.zeros((rows,cols),dtype = np.complex128)
    for i in range(rows):
        result[i,:] = engine.transform(plane[i, :])
    for j in range(cols):
        result[:,j] = engine.transform(result[:,j])
    return result
    #one dft takes o(N2), for N rows, total dft cost= o(N3), same for N columns

def inverse_2d(spectrum, engine):
    """
    2D inverse transform, the same way round. Shape is preserved.
    """
    # TODO: implement this function
    spectrum = np.asarray(spectrum, dtype = np.complex128)
    rows,cols = spectrum.shape
    result = np.zeros((rows,cols),dtype = np.complex128)
    for i in range(rows):
        result[i,:] = engine.inverse(spectrum[i,:])
    for j in range(cols):
        result[:,j] = engine.inverse(result[:,j])
    return result

def convolve_plane(plane, kernel, engine, circular=False):
    """
    Convolve one 2D plane with ``kernel`` through the frequency domain.

    Returns an array the SAME size as the input plane.

    circular=False (the normal case -- linear convolution):
        The full convolution of an (H, W) image with a (kh, kw) kernel is
        (H+kh-1, W+kw-1). Zero-pad both arrays to at least that size before
        transforming -- with FFTTransformer, pad further, up to a power of two
        in each dimension (every engine exposes a ``name`` attribute, so
        ``engine.name == "fft"`` tells you which rule applies). Multiply the
        two spectra, inverse-transform, take the real part, then crop the
        (H, W) window that corresponds to the original pixels: rows
        kh//2 .. kh//2+H-1 and columns kw//2 .. kw//2+W-1
        (the kernel sits at the origin of the padded array, so the result is
        offset by half the kernel -- forget this and your image comes out
        shifted diagonally).

    circular=True (the deliberate mistake -- see the specification):
        Transform at exactly (H, W) with no padding at all, with the kernel
        wrapped around the origin (np.roll is allowed -- it is not a
        transform). The output is the CIRCULAR convolution: content that
        should have fallen off one edge reappears on the opposite edge. The
        provided images are 256x256 and 512x512, so this path works with the
        radix-2 FFT directly.

    Parameters
    ----------
    plane : 2D numpy array of float, values in [0, 1]
    kernel : 2D numpy array of float, sums to 1
    engine : DFTAnalyzer or FFTTransformer
    circular : bool

    Returns
    -------
    numpy.ndarray of float64, same shape as ``plane``
    """
    # TODO: implement this function
    plane = np.asarray(plane, dtype = np.float64)
    kernel = np.asarray(kernel, dtype = np.float64)
    
    H,W = plane.shape
    kh, kw = kernel.shape
    if circular:
        padded_plane =  plane.astype(np.complex128)
        padded_kernel = np.zeros((H,W),dtype = np.complex128)
        padded_kernel[:kh, :kw] = kernel
        
        padded_kernel = np.roll(padded_kernel,-(kh//2),axis = 0)
        padded_kernel = np.roll(padded_kernel,-(kw//2),axis=1)
        spectrum_plane = transform_2d(padded_plane,engine)
        spectrum_kernel = transform_2d(padded_kernel,engine)
        spectrum_result = spectrum_kernel * spectrum_plane
        result = inverse_2d(spectrum_result, engine)
        return np.real(result)
    
    H_full = H + kh -1
    W_full = W + kw -1
    if (engine.name == "fft"):
        H_transformed = next_power_of_two(H_full)
        W_transformed = next_power_of_two(W_full)
    else:
        H_transformed = H_full
        W_transformed = W_full
        
    padded_plane = np.zeros((H_transformed,W_transformed),dtype = np.complex128)
    padded_kernel = np.zeros((H_transformed,W_transformed),dtype = np.complex128)
    padded_plane[:H,:W] = plane
    padded_kernel[:kh,:kw] = kernel
    
    spectrum_plane = transform_2d(padded_plane,engine)
    spectrum_kernel = transform_2d(padded_kernel,engine)
    
    spectrum_result = spectrum_plane * spectrum_kernel
    result = inverse_2d(spectrum_result,engine).real
    
    orig_row = kh//2
    orig_col = kw//2
    final = result[orig_row: orig_row+H, orig_col: orig_col+W]
    return final

def convolve_image(image, kernel, engine, circular=False):
    """
    Apply convolve_plane to a whole image.

    A grayscale image is (H, W); a colour image is (H, W, 3) and each colour
    plane is convolved independently, then stacked back together.
    """
    # TODO: implement this function
    if (image.ndim == 2):
        return convolve_plane(image, kernel, engine, circular)
    rows,cols,cnl = image.shape
    result = np.zeros((rows,cols,cnl),dtype = np.float64)
    for p in range(cnl):
        result[:,:,p] = convolve_plane(image[:,:,p],kernel,engine,circular)
    return result


def convolve_plane_direct(plane, kernel):
    """
    Spatial convolution, written out literally, as the correctness oracle and
    the third benchmark curve.

        out[r, c] = sum_i sum_j  plane[r + kh//2 - i, c + kw//2 - j] * kernel[i, j]

    with out-of-range pixels treated as zero. Four nested loops, O(N^2 K^2),
    no NumPy vectorisation -- this one is meant to be slow and obviously
    correct. It is never applied to a full 512x512 image (see run_single).
    """
    # TODO: implement this function
    H, W = plane.shape
    kh,kw = kernel.shape
    out = np.zeros((H,W),dtype = np.float64)
    for r in range(H):
        for c in range(W):
            for i in range(kh):
                for j in range(kw):
                    r_plane = r + kh//2 - i 
                    c_plane = c + kw//2 - j
                    if (0<=r_plane< H ) and (0<=c_plane < W):
                        out[r,c] += (plane[r_plane, c_plane]*kernel[i,j])
    return out


def run_single(path, kernel_name, param, engine_name, out_dir, gray=False):
    """
    Blur one image and write the required outputs.

    Build the kernel with image_utils.make_kernel:
        bokeh    -> make_kernel("bokeh", radius=param)
        gaussian -> make_kernel("gaussian", size=param)
        box      -> make_kernel("box", size=param)
        motion   -> make_kernel("motion", length=param, angle=30.0)

    Must produce, inside ``out_dir``:
      blurred.png     -- the linear (zero-padded) convolution
      wraparound.png  -- the same blur computed circularly, with no padding
      kernel.png      -- image_utils.save_kernel_preview of the kernel
      comparison.png  -- image_utils.save_comparison of original / blurred /
                         wraparound, side by side
      report.txt      -- image path and size, kernel name and size, engine,
                         the linear-convolution size, the transform size you
                         actually used, and the verification result. It is
                         written by your code; there is no separate write-up
                         to hand in.

    Verification: convolve the top-left 64x64 corner of the image (first colour
    plane, if colour) both ways -- convolve_plane and convolve_plane_direct --
    and report max |spectral - direct|. It should be ~1e-15, and anything above
    1e-9 is a bug, not rounding.
    """
    # TODO: implement this function
    os.makedirs(out_dir, exist_ok = True)
    
    image = load_image(path, as_gray=gray)
    
    if (kernel_name == "bokeh"):
        kernel = make_kernel("bokeh",radius = param)
    elif (kernel_name == "gaussian"):
        kernel = make_kernel("gaussian", size = param)
    elif (kernel_name == "box"):
        kernel = make_kernel("box" , size = param)
    elif (kernel_name == "motion"):
        kernel = make_kernel("motion", length = param, angle = 30.0)
    else:
        raise ValueError("kernel not known")
    
    if engine_name == "dft":
        engine = DFTAnalyzer()
    elif engine_name == "fft":
        engine = FFTTransformer()
    elif engine_name == "arbitrary":
        engine = ArbitraryLengthFFT()
    else:
        raise ValueError("engine not known")
    
    blurred = convolve_image(image, kernel, engine, circular=False)
    wraparound = convolve_image(image, kernel, engine, circular=True)

    blurred_path = os.path.join(out_dir, "blurred.png")
    wraparound_path = os.path.join(out_dir, "wraparound.png")
    kernel_path = os.path.join(out_dir, "kernel.png")
    comparison_path = os.path.join(out_dir, "comparison.png")

    save_image(blurred, blurred_path)
    save_image(wraparound, wraparound_path)
    save_kernel_preview(kernel,kernel_path,title=kernel_name +" kernel")
    images = [image, blurred, wraparound]
    labels = ["original","linear convolution (zero-padded)","circular convolution (no padding)"]
    save_comparison(images,labels,comparison_path)

    if image.ndim == 2:
        crop = image[:64, :64]
    else:
        crop = image[:64, :64, 0]
        
    direct = convolve_plane_direct(crop, kernel)
    spectral = convolve_plane(crop, kernel, engine, circular=False)
    difference = np.abs(spectral - direct)
    max_error = float(np.max(difference))

    if max_error <= 1e-9:
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"

    H, W = image.shape[:2]
    kh, kw = kernel.shape

    H_linear = H + kh - 1
    W_linear = W + kw - 1

    if engine.name == "dft":
        transform_H = H_linear
        transform_W = W_linear
    else:
        transform_H = next_power_of_two(H_linear)
        transform_W = next_power_of_two(W_linear)

    if image.ndim == 2:
        image_type = "gray"
    else:
        image_type = "RGB"

    report_lines = [
        "Task B -- 2D convolution through the frequency domain",
        "image               : %s  (%d x %d, %s)" % (path, H, W, image_type),
        "kernel              : %s  (%d x %d)" % (kernel_name, kh, kw),
        "engine              : %s" % engine_name,
        "linear-conv size    : %d x %d" % (H_linear, W_linear),
        "transform size      : %d x %d" % (transform_H, transform_W),
        "max |spectral - direct| on 64x64 crop : %.3e" % max_error,
        "verification        : %s" % verdict,
    ]

    report_path = os.path.join(out_dir, "report.txt")
    write_report(report_path, report_lines)
    return blurred, wraparound, kernel


# ---------------------------------------------------------------------------
# PROVIDED -- run_benchmark is already written. It calls your convolve_plane
# and convolve_plane_direct, so it starts working as soon as those are
# correct. You do not need to modify anything below (though you may extend
# it).
# ---------------------------------------------------------------------------
IMAGE_SIZES = [16, 32, 64, 128, 256, 512]
KERNEL_RADII = [1, 3, 7, 15, 31]
BENCH_RADIUS = 7            # kernel used for the growing-image study
BENCH_SIZE = 256            # image crop used for the growing-kernel study
TIME_BUDGET = 8.0           # stop a sweep once one measurement exceeds this


def run_benchmark(path, out_dir):
    """
    Two timing studies, two plots, both on one grayscale plane:

      1. growing image, fixed kernel   -> runtime_vs_image_size.png
      2. growing kernel, fixed image   -> runtime_vs_kernel_size.png

    plus both timing tables in report.txt. Each sweep stops early once a
    single measurement exceeds TIME_BUDGET seconds, so a slow machine simply
    produces a shorter curve rather than hanging.
    """
    full = load_image(path, as_gray=True)

    def sweep(label, make_call, points):
        """points: list of (x_value, zero-argument-callable-factory input)."""
        xs, ys = [], []
        print("%s:" % label)
        for x, arg in points:
            seconds = time_best(make_call(arg), repeats=1)
            xs.append(x)
            ys.append(seconds)
            print("  %8s   %9.4f s" % (x, seconds))
            if seconds > TIME_BUDGET:
                print("  (stopping this curve -- over the time budget)")
                break
        return xs, ys

    # ---- study 1: fixed kernel, growing image
    kernel = make_kernel("bokeh", radius=BENCH_RADIUS)
    crops = [(n, full[:n, :n].copy()) for n in IMAGE_SIZES]

    size_series = {}
    size_series["Naive DFT (row-column)"] = sweep(
        "naive DFT", lambda img: (lambda: convolve_plane(img, kernel, DFTAnalyzer())), crops)
    size_series["Radix-2 FFT (row-column)"] = sweep(
        "radix-2 FFT", lambda img: (lambda: convolve_plane(img, kernel, FFTTransformer())), crops)
    size_series["Direct spatial convolution"] = sweep(
        "direct spatial", lambda img: (lambda: convolve_plane_direct(img, kernel)), crops)

    size_plot = os.path.join(out_dir, "runtime_vs_image_size.png")
    plot_runtime_curve(size_series, size_plot,
                       title="Task B: %d x %d blur of an N x N image" % kernel.shape,
                       xlabel="image side length N (pixels)",
                       references=("n3", "n2"))

    # ---- study 2: fixed image, growing kernel
    image = full[:BENCH_SIZE, :BENCH_SIZE].copy()
    kernels = [(make_kernel("bokeh", radius=r).shape[0], make_kernel("bokeh", radius=r))
               for r in KERNEL_RADII]

    kernel_series = {}
    kernel_series["Direct spatial convolution"] = sweep(
        "direct spatial", lambda k: (lambda: convolve_plane_direct(image, k)), kernels)
    kernel_series["Radix-2 FFT (row-column)"] = sweep(
        "radix-2 FFT", lambda k: (lambda: convolve_plane(image, k, FFTTransformer())), kernels)

    kernel_plot = os.path.join(out_dir, "runtime_vs_kernel_size.png")
    plot_runtime_curve(kernel_series, kernel_plot,
                       title="Task B: %d x %d image, growing kernel" % image.shape,
                       xlabel="kernel side length K (pixels)",
                       references=("n2",))

    write_report(os.path.join(out_dir, "report.txt"),
                 ["Task B -- runtime benchmark", "",
                  "Study 1: fixed %d x %d kernel, growing image" % kernel.shape, ""]
                 + timing_table_lines(size_series, size_label="N")
                 + ["", "plot: %s" % os.path.basename(size_plot), "",
                    "Study 2: fixed %d x %d image, growing kernel" % image.shape, ""]
                 + timing_table_lines(kernel_series, size_label="K")
                 + ["", "plot: %s" % os.path.basename(kernel_plot)])
    print("wrote", size_plot, "and", kernel_plot)


def main():
    ap = argparse.ArgumentParser(description="2D convolution by DFT/FFT")
    ap.add_argument("image", help="path to the input image")
    ap.add_argument("--kernel", default="bokeh",
                    choices=["bokeh", "gaussian", "box", "motion"])
    ap.add_argument("--param", type=float, default=9,
                    help="bokeh radius / gaussian size / box size / motion length")
    ap.add_argument("--engine", default="fft", choices=["dft", "fft", "arbitrary"])
    ap.add_argument("--gray", action="store_true", help="process as grayscale")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the timing study instead of a single blur")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.benchmark:
        run_benchmark(args.image, args.out_dir)
    else:
        run_single(args.image, args.kernel, args.param, args.engine,
                   args.out_dir, gray=args.gray)


if __name__ == "__main__":
    main()
