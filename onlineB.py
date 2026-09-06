"""
Lab evaluation template: swap the Fourier magnitudes and phases of two images.

Complete TODO 1--3. Do not modify the original offline files.

Run with the supplied images:

    python onlineB.py --out-dir outputs/lab_phase_swap_student
"""

import argparse
import os

import numpy as np

from image_conv import inverse_2d, transform_2d
from image_utils import load_image, save_comparison, save_image
from io_utils import write_report
from transforms import ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer


def unit_phase(spectrum, epsilon=1e-12):
    """Return unit-magnitude complex values carrying only spectral phase."""
    # TODO 1 (student): retain only the phase of every frequency bin. A bin
    # whose magnitude is greater than epsilon is safe to normalize.
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    magnitude = np.abs(spectrum)

    # Create result with the neutral phase 1+0j in every position. Create a
    # Boolean mask for positions where magnitude > epsilon. At those masked
    # positions, divide spectrum by magnitude; leave all other positions at
    # 1+0j so that division by zero never occurs. Then return result.
    # result =
    # reliable =
    # result[reliable] =
    # return
    # m, n= spectrum.shape
    # result = np.zeros((m,n),dtype=np.complex128)
    # for i in range(m):
    #     for j in range(n):
    #         result[i,j]=1 + 0j
    result = np.ones_like(spectrum,dtype = np.complex128)
    reliable = magnitude > epsilon
    result[reliable] = spectrum[reliable]/magnitude[reliable]
    return result
    #raise NotImplementedError("TODO 1: extract unit phase safely")


def swap_plane_spectra(plane_a, plane_b, engine):
    """Return magnitude(A)+phase(B) and magnitude(B)+phase(A) images."""
    # TODO 2 (student): transform both planes, construct the two exchanged
    # spectra, inverse-transform them, and return their real components.
    plane_a = np.asarray(plane_a, dtype=np.float64)
    plane_b = np.asarray(plane_b, dtype=np.float64)
    if plane_a.ndim != 2 or plane_a.shape != plane_b.shape:
        raise ValueError("the two planes must have the same 2D shape")
    spectrum_a = transform_2d(plane_a, engine)
    spectrum_b = transform_2d(plane_b, engine)

    # Construct swapped_ab using abs(spectrum_a) as its magnitude and
    # unit_phase(spectrum_b) as its phase. Construct swapped_ba in the
    # opposite direction using abs(spectrum_b) and unit_phase(spectrum_a).
    # swapped_ab =
    # swapped_ba =
    ab_mag = np.abs(spectrum_a)
    # ab_phase = np.zeros(len(ab_mag),dtype=np.float64)
    # intem_ab = unit_phase(spectrum_a)
    ab_phase = unit_phase(spectrum_b)
    # ab_phase[:len(intem_ab)]=intem_ab
    swapped_ab = ab_mag * (ab_phase)
    ba_mag = np.abs(spectrum_b)
    #ba_phase = np.zeros(len(ba_mag),dtype = np.float64)
    ba_phase = unit_phase(spectrum_a)
    #intem_ba=unit_phase(spectrum_a)
    #ba_phase[:len(intem_ba)]=intem_ba
    swapped_ba = ba_mag * (ba_phase)
    # Apply inverse_2d to each swapped spectrum. Numerical roundoff may leave
    # tiny imaginary values, so retain the .real component of each result.
    # result_ab =
    # result_ba =
    # return
    result_ab = inverse_2d(swapped_ab,engine).real
    result_ba = inverse_2d(swapped_ba,engine).real
    return result_ab, result_ba
    # raise NotImplementedError("TODO 2: exchange magnitude and phase")


def swap_images(image_a, image_b, engine):
    """Apply the magnitude-phase swap to matching grayscale or RGB images."""
    # TODO 3 (student): dispatch one grayscale plane or three RGB planes and
    # return both reconstructed images with their original shape.
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = np.asarray(image_b, dtype=np.float64)
    if image_a.shape != image_b.shape:
        raise ValueError("the two images must have the same shape")
    if image_a.ndim == 2:
        # For two-dimensional grayscale inputs, call swap_plane_spectra once
        # using the complete images and directly return its pair of results.
        # return
        return swap_plane_spectra(image_a,image_b,engine)
    if image_a.ndim == 3 and image_a.shape[2] == 3:
        # For RGB inputs, call swap_plane_spectra on corresponding channel c
        # from the two images for c=0,1,2. Each call returns an (ab, ba) pair.
        # Stack all ab planes along axis 2, do the same for all ba planes, and
        # return the two reconstructed RGB arrays.
        # pairs =
        # result_ab =
        # result_ba =
        # return
        ab_planes=[]
        ba_planes=[]
        for channels in range(3):
            ab, ba = swap_plane_spectra(image_a[:,:,channels],image_b[:,:,channels],engine)
            ab_planes.append(ab)
            ba_planes.append(ba)
        result_ab = np.stack(ab_planes,axis=2)
        result_ba = np.stack(ba_planes, axis=2)
        return result_ab, result_ba
        #raise NotImplementedError("TODO 3: process RGB images")
    # raise ValueError("images must be grayscale or RGB")


def _make_engine(name):
    """Provided command-line engine selection."""
    if name == "dft":
        return DFTAnalyzer()
    if name == "fft":
        return FFTTransformer()
    if name == "arbitrary":
        return ArbitraryLengthFFT()
    raise ValueError("unknown engine: %r" % name)


def _normalise_for_display(array):
    """Provided robust contrast scaling for a possibly signed result."""
    array = np.asarray(array, dtype=np.float64)
    low, high = np.percentile(array, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _plane_verification(result, magnitude_source, phase_source, engine):
    """Provided independent magnitude, phase, and energy checks."""
    actual = transform_2d(result, engine)
    magnitude_target = transform_2d(magnitude_source, engine)
    phase_target = transform_2d(phase_source, engine)

    scale = max(1.0, float(np.max(np.abs(magnitude_target))))
    magnitude_error = float(np.max(
        np.abs(np.abs(actual) - np.abs(magnitude_target)))) / scale

    phase_size = np.abs(phase_target)
    actual_size = np.abs(actual)
    threshold = 1e-10 * max(1.0, float(np.max(phase_size)))
    reliable = (phase_size > threshold) & (actual_size > threshold)
    if np.any(reliable):
        actual_phase = actual[reliable] / actual_size[reliable]
        target_phase = phase_target[reliable] / phase_size[reliable]
        phase_error = float(np.max(np.abs(actual_phase - target_phase)))
    else:
        phase_error = 0.0

    source_energy = float(np.sum(np.square(magnitude_source)))
    result_energy = float(np.sum(np.square(result)))
    energy_error = abs(result_energy - source_energy) / max(1.0, source_energy)
    return magnitude_error, phase_error, energy_error


def _verification(result, magnitude_source, phase_source, engine):
    """Verify every grayscale or colour plane and return worst-case errors."""
    if result.ndim == 2:
        return _plane_verification(result, magnitude_source, phase_source,
                                   engine)
    errors = [_plane_verification(result[:, :, c], magnitude_source[:, :, c],
                                  phase_source[:, :, c], engine)
              for c in range(3)]
    return tuple(max(values) for values in zip(*errors))


def run(image_a_path, image_b_path, engine_name, out_dir, color=False):
    """Provided runner: construct, save, and verify both swapped images."""
    engine = _make_engine(engine_name)
    image_a = load_image(image_a_path, as_gray=not color)
    image_b = load_image(image_b_path, as_gray=not color)
    result_ab, result_ba = swap_images(image_a, image_b, engine)

    errors_ab = _verification(result_ab, image_a, image_b, engine)
    errors_ba = _verification(result_ba, image_b, image_a, engine)
    worst_magnitude = max(errors_ab[0], errors_ba[0])
    worst_phase = max(errors_ab[1], errors_ba[1])
    worst_energy = max(errors_ab[2], errors_ba[2])
    verdict = ("MATCH" if max(worst_magnitude, worst_phase, worst_energy)
               <= 1e-9 else "MISMATCH")

    display_ab = _normalise_for_display(result_ab)
    display_ba = _normalise_for_display(result_ba)
    os.makedirs(out_dir, exist_ok=True)
    label_a = os.path.splitext(os.path.basename(image_a_path))[0]
    label_b = os.path.splitext(os.path.basename(image_b_path))[0]
    ab_path = os.path.join(out_dir, "magnitude_%s_phase_%s.png" %
                           (label_a, label_b))
    ba_path = os.path.join(out_dir, "magnitude_%s_phase_%s.png" %
                           (label_b, label_a))
    save_image(display_ab, ab_path)
    save_image(display_ba, ba_path)
    save_comparison(
        [image_a, image_b, display_ab, display_ba],
        [label_a, label_b, "magnitude: %s, phase: %s" % (label_a, label_b),
         "magnitude: %s, phase: %s" % (label_b, label_a)],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Fourier magnitude-phase swap, engine=%s" % engine_name,
    )

    write_report(os.path.join(out_dir, "report.txt"), [
        "Lab evaluation -- Fourier magnitude-phase swap",
        "image A              : %s" % image_a_path,
        "image B              : %s" % image_b_path,
        "image shape          : %s" % (image_a.shape,),
        "engine               : %s" % engine_name,
        "worst relative magnitude error : %.3e" % worst_magnitude,
        "worst phase-vector error       : %.3e" % worst_phase,
        "worst relative energy error    : %.3e" % worst_energy,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict)
    print("magnitude error %.3e, phase error %.3e, energy error %.3e" %
          (worst_magnitude, worst_phase, worst_energy))
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("magnitude-phase reconstruction failed verification")
    return result_ab, result_ba


def main():
    parser = argparse.ArgumentParser(description="Swap image Fourier magnitude and phase")
    parser.add_argument("--image-a", default="images/sunset512.png")
    parser.add_argument("--image-b", default="images/skyline512.png")
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"],
                        default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default="outputs/lab_phase_swap_student")
    args = parser.parse_args()
    run(args.image_a, args.image_b, args.engine, args.out_dir,
        color=args.color)


if __name__ == "__main__":
    main()
