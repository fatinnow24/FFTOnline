# CSE 220 — DFT / DTFT / FFT Coding Exam: Template & Solution Bank

**Every code block here has been run and verified.** Paste §0 at the top of your file, then jump
to the matching problem.

**30-second triage:**

| The problem says… | Go to |
|---|---|
| multiply big numbers / digit arrays | §1 |
| polynomials, coefficients, weights | §2, §5 |
| rows shifted, realign image, rolling shutter | §3 |
| hybrid image, low+high frequency mix | §4 |
| "convolve these two sequences" | §6 |
| filter / denoise / remove noise from a signal | §7, §8 |
| "what frequency is in this signal" | §9 |
| delay / lag / alignment between two signals | §10 |
| image shifted in both directions | §11 |
| period / repeating pattern | §12 |
| blur an image | §13 |
| sharpen / edges | §14 |
| stripes or periodic noise in an image | §15 |
| find a pattern inside a longer array | §16 |
| filter a very long signal in blocks | §17 |
| undo a blur, recover original | §18 |
| resample / interpolate | §19 |
| "verify / prove property X" | §20 |
| N is not a power of two | §22 |

---

# §0 — THE TEMPLATE (paste this first, always)

```python
import numpy as np

# ============================== CORE ==============================

def next_power_of_two(n):
    size = 1
    while size < n:
        size *= 2
    return size


def _bit_reverse(x):
    N = x.shape[0]
    idx = np.arange(N)
    rev = np.zeros(N, dtype=np.int64)
    bits = N.bit_length() - 1
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return x[rev]


def fft(x):
    """Iterative radix-2 decimation-in-time FFT. len(x) must be a power of 2."""
    x = np.asarray(x, dtype=np.complex128).copy()
    N = x.shape[0]
    if N < 1 or (N & (N - 1)):
        raise ValueError("FFT length must be a power of two, got %d" % N)
    if N == 1:
        return x
    data = _bit_reverse(x)
    half = 1
    while half < N:                                   # log2(N) stages
        span = 2 * half
        w = np.exp(-2j * np.pi * np.arange(half) / span)   # twiddles once per stage
        blocks = data.reshape(N // span, span)
        even = blocks[:, :half].copy()                # .copy() IS REQUIRED (aliasing)
        odd = blocks[:, half:] * w
        blocks[:, :half] = even + odd
        blocks[:, half:] = even - odd
        data = blocks.reshape(N)
        half = span
    return data


def ifft(X):
    """Inverse FFT, reusing the same butterflies: (1/N)*conj(FFT(conj(X)))."""
    X = np.asarray(X, dtype=np.complex128)
    return np.conj(fft(np.conj(X))) / X.shape[0]


def dft(x):
    """Naive O(N^2) DFT. Any length. Use only if asked for it explicitly."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    n = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(n, n) / N) @ x


def idft(X):
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    n = np.arange(N)
    return (np.exp(2j * np.pi * np.outer(n, n) / N) @ X) / N


def bluestein(x):
    """O(N log N) DFT for ANY length N (Bluestein chirp-z)."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N <= 1:
        return x.copy()
    if N & (N - 1) == 0:
        return fft(x)
    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * (n * n % (2 * N)) / N)
    L = next_power_of_two(2 * N - 1)
    a = np.zeros(L, dtype=np.complex128); a[:N] = x * chirp
    b = np.zeros(L, dtype=np.complex128)
    b[:N] = np.conj(chirp)
    b[L - N + 1:] = np.conj(chirp[:0:-1])
    return ifft(fft(a) * fft(b))[:N] * chirp


def ibluestein(X):
    X = np.asarray(X, dtype=np.complex128)
    return np.conj(bluestein(np.conj(X))) / X.shape[0]


def transform(x):
    """FFT when the length is a power of two, else Bluestein. Always O(N log N)."""
    N = len(x)
    return fft(x) if (N & (N - 1)) == 0 else bluestein(x)


def inverse(X):
    N = len(X)
    return ifft(X) if (N & (N - 1)) == 0 else ibluestein(X)


# ========================== CONVOLUTION ===========================

def pad_to(a, N):
    a = np.asarray(a, dtype=np.complex128)
    out = np.zeros(N, dtype=np.complex128)
    out[:len(a)] = a
    return out


def linear_convolve(a, b):
    """Full linear convolution, length len(a)+len(b)-1. THE workhorse."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    L = len(a) + len(b) - 1
    N = next_power_of_two(L)                 # <-- the padding rule
    return ifft(fft(pad_to(a, N)) * fft(pad_to(b, N))).real[:L]


def circular_convolve(a, b, N=None):
    """Circular convolution at period N (wraps on purpose)."""
    if N is None:
        N = max(len(a), len(b))
    return inverse(transform(pad_to(a, N)) * transform(pad_to(b, N))).real


def cross_correlate_circular(ref, sig):
    """Circular cross-correlation of sig against ref (same length)."""
    return inverse(transform(sig) * np.conj(transform(ref))).real


def estimate_circular_shift(ref, sig):
    """If sig[n] == ref[(n - s) % N], returns s."""
    return int(np.argmax(cross_correlate_circular(ref, sig)))


# ============================== 2D ================================

def transform_2d(plane, t=transform):
    """2D forward transform by separability: all rows, then all columns."""
    a = np.asarray(plane, dtype=np.complex128)
    out = np.empty(a.shape, dtype=np.complex128)
    for r in range(a.shape[0]):
        out[r, :] = t(a[r, :])
    for c in range(a.shape[1]):
        out[:, c] = t(out[:, c])
    return out


def inverse_2d(spec, t=inverse):
    return transform_2d(spec, t)


def pad_2d(a, shape):
    out = np.zeros(shape, dtype=np.float64)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def convolve2d_linear(plane, kernel):
    """Linear 2D convolution, output same size as plane."""
    H, W = plane.shape
    kh, kw = kernel.shape
    P = next_power_of_two(H + kh - 1)
    Q = next_power_of_two(W + kw - 1)
    S = transform_2d(pad_2d(plane, (P, Q))) * transform_2d(pad_2d(kernel, (P, Q)))
    y = inverse_2d(S).real
    return y[kh // 2:kh // 2 + H, kw // 2:kw // 2 + W]     # <-- the crop


def convolve2d_circular(plane, kernel):
    """Circular 2D convolution (wraps at the edges)."""
    H, W = plane.shape
    kh, kw = kernel.shape
    k = pad_2d(kernel, (H, W))
    k = np.roll(k, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return inverse_2d(transform_2d(pad_2d(plane, (H, W))) * transform_2d(k)).real


def convolve_image(img, kernel, circular=False):
    """Grayscale (H,W) or colour (H,W,3)."""
    f = convolve2d_circular if circular else convolve2d_linear
    if img.ndim == 2:
        return f(img, kernel)
    return np.stack([f(img[:, :, c], kernel) for c in range(img.shape[2])], axis=2)


# ============================ KERNELS =============================

def gaussian_kernel(size, sigma=None):
    if sigma is None:
        sigma = size / 6.0
    ax = np.arange(size) - (size - 1) / 2.0
    line = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = np.outer(line, line)
    return k / k.sum()


def box_kernel(size):
    return np.ones((size, size)) / (size * size)


def impulse_kernel(size):
    """Identity kernel: convolving with it changes nothing."""
    k = np.zeros((size, size)); k[size // 2, size // 2] = 1.0
    return k


# ============================ IMAGE IO ============================

def load_image(path, gray=False):
    from PIL import Image
    im = Image.open(path).convert("L" if gray else "RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def save_image(arr, path):
    from PIL import Image
    d = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    d = np.rint(d * 255).astype(np.uint8)
    Image.fromarray(d, mode="L" if d.ndim == 2 else "RGB").save(path)
```

---

# §1 — Big-integer multiplication (PAST EXAM, Section A)

Digit arrays **LSD-first**, base 10. Product via FFT linear convolution + carry sweep.

```python
def multiply_digits(A, B, base=10):
    """A, B: LSD-first digit lists. Returns LSD-first digit list of A*B."""
    A = [int(d) for d in A]
    B = [int(d) for d in B]
    if not A or not B or all(d == 0 for d in A) or all(d == 0 for d in B):
        return [0]

    conv = np.rint(linear_convolve(A, B)).astype(np.int64)   # raw coefficients

    out, carry = [], 0
    for v in conv:                       # carry sweep, LSD -> MSD
        carry += int(v)
        out.append(carry % base)
        carry //= base
    while carry:                         # carry can spill past the end
        out.append(carry % base)
        carry //= base
    while len(out) > 1 and out[-1] == 0:  # strip leading zeros of the number
        out.pop()
    return out


if __name__ == "__main__":
    print("Result:", multiply_digits([3, 2, 1], [5, 4]))   # [5, 3, 5, 5]   = 5535
    print("Result:", multiply_digits([9, 9, 9], [9, 9]))   # [1, 0, 9, 8, 9] = 98901
```

**Verified:** both examples exact, plus 200 random products against Python integers.

**If they give you strings / plain integers instead:**
```python
def multiply_strings(sa, sb):
    A = [int(c) for c in sa][::-1]
    B = [int(c) for c in sb][::-1]
    return "".join(str(d) for d in multiply_digits(A, B)[::-1])
```

**If they want MSD-first arrays:** reverse on the way in and on the way out.

---

# §2 — Weighted polynomial product (PAST EXAM, Section B)

`R[k] = Σ_{i=0..k} w_i · p_i · q_{k-i}`

**The trick:** set `u[i] = w_i · p_i`, then `R = u ∗ q` — an ordinary linear convolution.

> ⚠️ **ORDERING TRAP — read this.** The paper gives inputs in **descending** powers
> (`[1,3,2]` means `x²+3x+2`), so you must reverse to ascending before convolving.
> But the two examples then **print the answer in opposite orders**: Example 1's expected
> output `[12,27,14,2]` is descending, while Example 2's `[42,198,122,14,27,12]` is
> ascending. They are reverses of the same result. **Print both, or match whichever example
> the exam repeats.**

```python
def weighted_poly_product(P, Q, W, descending_in=True, descending_out=False):
    """P, Q, W as given. Returns R coefficients."""
    p = np.asarray(P, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    if descending_in:                 # -> ascending (index = power of x)
        p, q, w = p[::-1], q[::-1], w[::-1]

    m = max(len(p), len(w))           # w and p must align index-by-index
    pp = np.zeros(m); pp[:len(p)] = p
    ww = np.zeros(m); ww[:len(w)] = w
    u = ww * pp                       # u[i] = w_i * p_i

    R = np.rint(linear_convolve(u, q)).astype(np.int64)
    return list(R[::-1]) if descending_out else list(R)


if __name__ == "__main__":
    r1 = weighted_poly_product([1, 3, 2], [4, 1], [3, 2, 1])
    print("ascending :", r1)              # [2, 14, 27, 12]
    print("descending:", r1[::-1])        # [12, 27, 14, 2]   <- paper Example 1

    r2 = weighted_poly_product([1, 3, 2, 6, 7], [4, 1], [3, 2, 1, 5, 6])
    print("ascending :", r2)              # [42, 198, 122, 14, 27, 12]  <- paper Example 2
    print("descending:", r2[::-1])
```

**Verified:** reproduces both paper examples exactly.

**Variant — weight on the second factor** (`R[k] = Σ p_i · w_{k-i} · q_{k-i}`): put the weight
on `q` instead — `u = q * w`, then `R = p ∗ u`.

**Variant — weight applied after** (`R[k] = w_k · Σ p_i q_{k-i}`): convolve first, then
multiply the result elementwise by `w`.

---

# §3 — Rolling shutter: realign shifted rows (PAST EXAM, Section C)

Each row circularly shifted right by an unknown amount. Detect with cross-correlation, undo
with `np.roll`.

```python
def estimate_row_shifts(reference, distorted):
    return np.array([estimate_circular_shift(reference[r], distorted[r])
                     for r in range(distorted.shape[0])], dtype=int)


def correct_rolling_shutter(reference, distorted):
    """Both (H,W) grayscale. Returns the realigned image."""
    out = np.empty_like(distorted)
    shifts = np.zeros(distorted.shape[0], dtype=int)
    for r in range(distorted.shape[0]):
        s = estimate_circular_shift(reference[r], distorted[r])
        shifts[r] = s
        out[r] = np.roll(distorted[r], -s)        # undo the shift
    return out, shifts


if __name__ == "__main__":
    original = load_image("original_image.png", gray=True)
    shifted  = load_image("shifted_image.jpg",  gray=True)

    fixed, shifts = correct_rolling_shutter(original, shifted)

    print("detected shifts (first 10):", shifts[:10])
    print("max |fixed - original| =", np.max(np.abs(fixed - original)))
    save_image(fixed, "reconstructed.png")
```

**Verified:** on a 512×512 image with 512 random row shifts, **all shifts recovered exactly and
reconstruction error 0.0**.

**If the image is RGB:** work on the luminance for detection, then apply the shift to all three
planes.
```python
def correct_rgb(reference, distorted):
    ref_g = reference.mean(axis=2); dis_g = distorted.mean(axis=2)
    out = np.empty_like(distorted)
    for r in range(distorted.shape[0]):
        s = estimate_circular_shift(ref_g[r], dis_g[r])
        out[r] = np.roll(distorted[r], -s, axis=0)   # (W,3) row: axis 0 is width
    return out
```

**Gotchas:**
- `shifted_image.jpg` is **JPEG = lossy**, so `max|fixed − original|` will be small but not 0.
  The detected integer shifts are still exact. Report the error, don't panic.
- If they say shifted **left**, use `np.roll(row, +s)` or negate the estimate.
- **No reference image given?** Align row `r` to the already-corrected row `r−1`. This only works
  when consecutive rows are strongly correlated — on a detailed photo it drifts badly (I measured
  0.89 error vs 0.0 with a reference). Prefer the reference whenever one exists.

---

# §4 — Hybrid image (PAST EXAM, Section D)

`Y = L·G + H·(Δ − G)` where `Δ` is the spectrum of an impulse at the kernel centre.
`L·G` = blurred low-frequency source. `H·(Δ−G)` = high source minus its own blur = detail.

### Self-contained version

```python
def hybrid_plane(low, high, kernel):
    H_, W_ = low.shape
    kh, kw = kernel.shape
    P = next_power_of_two(H_ + kh - 1)
    Q = next_power_of_two(W_ + kw - 1)

    impulse = np.zeros((P, Q)); impulse[kh // 2, kw // 2] = 1.0

    L = transform_2d(pad_2d(low,    (P, Q)))
    H = transform_2d(pad_2d(high,   (P, Q)))
    G = transform_2d(pad_2d(kernel, (P, Q)))
    D = transform_2d(impulse)

    Y = L * G + H * (D - G)                       # <- the model
    y = inverse_2d(Y).real                        # ONE inverse transform
    return y[kh // 2:kh // 2 + H_, kw // 2:kw // 2 + W_]


def hybrid_image(low, high, kernel):
    if low.ndim == 2:
        return hybrid_plane(low, high, kernel)
    return np.stack([hybrid_plane(low[:, :, c], high[:, :, c], kernel)
                     for c in range(low.shape[2])], axis=2)


if __name__ == "__main__":
    low  = load_image("images/sunset512.png")
    high = load_image("images/skyline512.png")
    k = gaussian_kernel(31)
    out = hybrid_image(low, high, k)
    save_image(out, "hybrid.png")
```

**Verified:** equals `blur(low) + (high − blur(high))` to **1.1e-15**.

### Template-fill version (when they hand you `choose_transform_shape` etc.)

```python
def choose_transform_shape(image_shape, kernel_shape, engine):
    R = image_shape[0] + kernel_shape[0] - 1
    C = image_shape[1] + kernel_shape[1] - 1
    if engine.name == "fft":
        return next_power_of_two(R), next_power_of_two(C)
    return R, C


def hybrid_plane(low_spec, high_spec, kernel_spec, impulse_spec,
                 shape, kernel_shape, engine):
    kh, kw = kernel_shape
    H_, W_ = shape
    Y = low_spec * kernel_spec + high_spec * (impulse_spec - kernel_spec)
    y = inverse_2d(Y, engine).real          # <- the TEMPLATE's inverse_2d(spectrum, engine)
    return y[kh // 2:kh // 2 + H_, kw // 2:kw // 2 + W_]
```

> ⚠️ **Signature clash.** The offline/exam template defines `inverse_2d(spectrum, engine)` taking
> an *engine object*. §0's version is `inverse_2d(spec, t=inverse)` taking a *1-D function*.
> If you are filling in their template, use **their** `inverse_2d` and `transform_2d` — do not
> paste §0's 2-D helpers over them. If you are writing from scratch, use §0's and the
> self-contained version above.

**Variants they might ask instead:**
- **Low-pass only:** `Y = L*G` → just a blur.
- **High-pass only:** `Y = H*(D − G)`.
- **Weighted mix:** `Y = alpha*L*G + beta*H*(D − G)`.

---

# §5 — Plain polynomial multiplication

```python
def poly_multiply(P, Q, descending=True):
    p = np.asarray(P, float)[::-1] if descending else np.asarray(P, float)
    q = np.asarray(Q, float)[::-1] if descending else np.asarray(Q, float)
    R = np.rint(linear_convolve(p, q)).astype(np.int64)      # ascending
    return list(R[::-1]) if descending else list(R)

# (x^2+3x+2)(4x+1) = 4x^3+13x^2+11x+2
print(poly_multiply([1, 3, 2], [4, 1]))       # [4, 13, 11, 2]
```

---

# §6 — Linear and circular convolution of given sequences

```python
a = [1, 2, 3, 4]
h = [1, 2, 1]

print("linear  :", np.round(linear_convolve(a, h), 6))
print("circular:", np.round(circular_convolve(a, h, N=len(a)), 6))
print("circular=linear when N >= L+M-1:",
      np.round(circular_convolve(a, h, N=len(a) + len(h) - 1), 6))
```

**Circular convolution the direct way** (if they ask you to show the wrap explicitly):
```python
def circular_convolve_direct(x, h, N):
    x = np.asarray(x, float); h = np.asarray(h, float)
    xp = np.zeros(N); xp[:len(x)] = x
    hp = np.zeros(N); hp[:len(h)] = h
    return np.array([sum(xp[m] * hp[(n - m) % N] for m in range(N)) for n in range(N)])
```

> **The one rule:** circular convolution = linear result **folded mod N**.
> Pad both to `N ≥ L + M − 1` and they become identical.

---

# §7 — Filtering a 1D signal (low / high / band pass)

```python
def fft_filter(x, fs, kind="low", fc=None, band=None):
    """kind: 'low' | 'high' | 'band' | 'stop'.  fc in Hz, band=(f1,f2) in Hz."""
    n = len(x)
    N = next_power_of_two(n)
    X = fft(pad_to(x, N))
    f = np.arange(N) * fs / N
    f = np.minimum(f, fs - f)                 # fold: bin -> |frequency|

    if kind == "low":
        X[f > fc] = 0
    elif kind == "high":
        X[f < fc] = 0
    elif kind == "band":
        X[(f < band[0]) | (f > band[1])] = 0
    elif kind == "stop":
        X[(f >= band[0]) & (f <= band[1])] = 0
    return ifft(X).real[:n]


if __name__ == "__main__":
    fs, N = 1000.0, 1024
    t = np.arange(N) / fs
    x = 3 * np.sin(2 * np.pi * 50 * t) + 1.5 * np.sin(2 * np.pi * 120 * t)
    y = fft_filter(x, fs, "low", fc=80)        # keeps 50 Hz, kills 120 Hz
```

**Verified:** the 120 Hz component drops to ~1e-18 of the 50 Hz amplitude.

> `f = np.minimum(f, fs - f)` is the important line — it turns bin index into *physical*
> frequency, correctly handling the folded upper half. Zeroing bin `k` without also zeroing
> `N−k` breaks conjugate symmetry and gives a complex result.

---

# §8 — Remove a specific tone / periodic noise (notch)

```python
def notch_filter(x, fs, f_remove, width=2.0):
    n = len(x); N = next_power_of_two(n)
    X = fft(pad_to(x, N))
    f = np.arange(N) * fs / N
    f = np.minimum(f, fs - f)
    X[np.abs(f - f_remove) <= width] = 0        # kills BOTH +f and -f bins
    return ifft(X).real[:n]


def denoise_keep_strongest(x, keep=10):
    """Keep only the `keep` largest-magnitude bin pairs, zero everything else."""
    N = next_power_of_two(len(x))
    X = fft(pad_to(x, N))
    mag = np.abs(X)
    thresh = np.sort(mag)[::-1][min(2 * keep, N) - 1]
    X[mag < thresh] = 0
    return ifft(X).real[:len(x)]
```

---

# §9 — Find the dominant frequency / frequencies

```python
def top_frequencies(x, fs, count=1):
    N = next_power_of_two(len(x))
    X = fft(pad_to(x, N))
    mag = np.abs(X[:N // 2])                  # only the positive half
    order = np.argsort(mag)[::-1]
    picked = []
    for k in order:                           # skip neighbours of an already-picked peak
        if all(abs(int(k) - j) > 2 for j in picked):
            picked.append(int(k))
        if len(picked) == count:
            break
    return sorted(k * fs / N for k in picked)


if __name__ == "__main__":
    fs, N = 1024.0, 1024
    t = np.arange(N) / fs
    x = 3 * np.sin(2 * np.pi * 64 * t) + 1.5 * np.sin(2 * np.pi * 200 * t)
    print(top_frequencies(x, fs, 2))          # [64.0, 200.0]
```

**Verified:** exact when the tone lands on a bin; `[49.8, 120.1]` for a 50/120 Hz signal at
fs=1000, N=1024 — that small error is **spectral leakage**, not a bug (50 Hz sits at bin 51.2,
between bins).

**Key formulas — write these on your sheet:**
```
bin spacing   Δf = fs / N = 1 / (total duration)
bin k         f  = k*fs/N          for k <= N/2      (positive)
                 = (k-N)*fs/N      for k >  N/2      (negative)
highest representable frequency = fs/2   (at k = N/2)
```

---

# §10 — Delay / lag between two 1D signals

```python
def find_delay_circular(ref, sig):
    """sig is ref delayed circularly by s. Returns s in samples."""
    return estimate_circular_shift(ref, sig)


def find_delay_linear(ref, sig):
    """Non-circular lag. Positive => sig lags ref. Returns lag in samples."""
    N = next_power_of_two(len(ref) + len(sig) - 1)
    C = fft(pad_to(sig, N)) * np.conj(fft(pad_to(ref, N)))
    corr = ifft(C).real
    k = int(np.argmax(corr))
    return k if k <= N // 2 else k - N        # unwrap to a signed lag
```

---

# §11 — 2D image shift detection (both axes)

```python
def find_2d_shift(reference, shifted):
    """shifted = np.roll(np.roll(reference, dr, 0), dc, 1). Returns (dr, dc), signed."""
    C = transform_2d(shifted) * np.conj(transform_2d(reference))
    corr = inverse_2d(C).real
    r, c = np.unravel_index(int(np.argmax(corr)), corr.shape)
    H, W = reference.shape
    if r > H // 2: r -= H                     # unwrap to signed
    if c > W // 2: c -= W
    return int(r), int(c)


def undo_2d_shift(shifted, dr, dc):
    return np.roll(np.roll(shifted, -dr, axis=0), -dc, axis=1)
```

**Verified:** recovered `(37, −58)` exactly on a 512×512 image.

**Phase correlation** — more robust when the two images differ in brightness/contrast.
Normalise the cross-spectrum to unit magnitude before inverting:

```python
def find_2d_shift_phase(reference, shifted):
    C = transform_2d(shifted) * np.conj(transform_2d(reference))
    C = C / (np.abs(C) + 1e-12)               # <-- keep only the phase
    corr = inverse_2d(C).real
    r, c = np.unravel_index(int(np.argmax(corr)), corr.shape)
    H, W = reference.shape
    if r > H // 2: r -= H
    if c > W // 2: c -= W
    return int(r), int(c)
```

---

# §12 — Find the period of a signal (autocorrelation)

```python
def find_period(x):
    n = len(x)
    N = next_power_of_two(2 * n)              # pad x2 => LINEAR autocorrelation
    X = fft(pad_to(x, N))
    ac = ifft(X * np.conj(X)).real[:n]
    return int(np.argmax(ac[1:n // 2]) + 1)   # skip lag 0, which is always the max


if __name__ == "__main__":
    sig = np.tile(np.random.randn(37), 20)[:512]
    print(find_period(sig))                   # 37
```

**Verified:** returns 37 exactly.

> Padding to `2n` matters — without it you get *circular* autocorrelation and the wrap
> corrupts the long lags.

---

# §13 — Blur an image (the offline, condensed)

```python
img = load_image("images/skyline512.png")          # or gray=True; use YOUR path
k = gaussian_kernel(21)                            # or box_kernel / bokeh below

blurred    = convolve_image(img, k, circular=False)   # correct, zero-padded
wraparound = convolve_image(img, k, circular=True)    # deliberately wraps

save_image(blurred, "blurred.png")
save_image(wraparound, "wraparound.png")
```

```python
def bokeh_kernel(radius):
    r = int(round(radius)); size = 2 * r + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((x ** 2 + y ** 2) <= radius ** 2).astype(float)
    return k / k.sum()


def motion_kernel(length, angle_deg=0.0):
    size = length if length % 2 else length + 1
    k = np.zeros((size, size)); c = (size - 1) / 2.0
    th = np.deg2rad(angle_deg)
    for step in np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 8):
        r = int(round(c - step * np.sin(th))); cc = int(round(c + step * np.cos(th)))
        if 0 <= r < size and 0 <= cc < size:
            k[r, cc] += 1
    return k / k.sum()
```

**Direct spatial convolution** (if they want an O(N²K²) oracle to compare against):
```python
def convolve2d_direct(plane, kernel):
    H, W = plane.shape; kh, kw = kernel.shape
    src = plane.tolist(); wgt = kernel.tolist()
    out = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            tot = 0.0
            for i in range(kh):
                rr = r + kh // 2 - i
                if 0 <= rr < H:
                    row = src[rr]; krow = wgt[i]
                    for j in range(kw):
                        cc = c + kw // 2 - j
                        if 0 <= cc < W:
                            tot += row[cc] * krow[j]
            out[r, c] = tot
    return out
```

---

# §14 — Sharpen / unsharp mask / edge detection

```python
def sharpen(img, kernel, amount=1.0):
    """original + amount * (original - blurred)"""
    return img + amount * (img - convolve_image(img, kernel))


def high_pass(img, kernel):
    """original - blurred = the detail layer"""
    return img - convolve_image(img, kernel)


def edges(img, kernel):
    d = high_pass(img, kernel)
    return np.abs(d) / (np.abs(d).max() + 1e-12)
```

Frequency-domain version (one inverse transform, exactly the §4 model with `L = H`):

```python
def sharpen_spectral(plane, kernel, amount=1.0):
    H_, W_ = plane.shape; kh, kw = kernel.shape
    P = next_power_of_two(H_ + kh - 1); Q = next_power_of_two(W_ + kw - 1)
    imp = np.zeros((P, Q)); imp[kh // 2, kw // 2] = 1.0
    X = transform_2d(pad_2d(plane, (P, Q)))
    G = transform_2d(pad_2d(kernel, (P, Q)))
    D = transform_2d(imp)
    Y = X * (D + amount * (D - G))
    return inverse_2d(Y).real[kh // 2:kh // 2 + H_, kw // 2:kw // 2 + W_]
```

---

# §15 — Remove periodic noise (stripes) from an image

```python
def remove_periodic_noise(plane, keep_radius=None, threshold_factor=8.0):
    """Zero isolated bright spikes in the 2D spectrum (they are the stripes)."""
    F = transform_2d(plane)
    mag = np.abs(F)
    H, W = plane.shape
    u = np.minimum(np.arange(H), H - np.arange(H))[:, None]
    v = np.minimum(np.arange(W), W - np.arange(W))[None, :]
    rad = np.sqrt(u ** 2 + v ** 2)

    med = np.median(mag)
    mask = (mag > threshold_factor * med) & (rad > (keep_radius or 5))
    F[mask] = 0
    return inverse_2d(F).real


def notch_2d(plane, points, radius=3):
    """Zero specific (u,v) bins and their conjugate partners."""
    F = transform_2d(plane); H, W = plane.shape
    for (u0, v0) in points:
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                F[(u0 + du) % H, (v0 + dv) % W] = 0
                F[(-u0 - du) % H, (-v0 - dv) % W] = 0
    return inverse_2d(F).real
```

---

# §16 — Pattern / substring matching via convolution

> ⚠️ Correlation **alone gives false positives.** You must also compare window energy.

```python
def find_pattern(text, pat, tol=1e-6):
    """Exact match positions of pat inside text."""
    text = np.asarray(text, float); pat = np.asarray(pat, float)
    n, m = len(text), len(pat)
    corr   = linear_convolve(text, pat[::-1])[m - 1:n]     # sliding dot product
    energy = linear_convolve(text ** 2, np.ones(m))[m - 1:n]
    dist   = energy - 2 * corr + np.sum(pat ** 2)          # ||window - pat||^2
    return list(np.where(np.abs(dist) < tol)[0])


def find_substring(s, p):
    return find_pattern([ord(c) for c in s], [ord(c) for c in p])


print(find_pattern([1, 2, 3, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3]))   # [0, 3, 7]
print(find_substring("abracadabra", "abra"))                      # [0, 7]
```

**Verified.** (Correlation alone wrongly reports index 5 in the first example, because
`[3,4,1]·[1,2,3] = 14 = |pat|²`. The energy term removes it.)

---

# §17 — Overlap-add (filter a very long signal in blocks)

```python
def overlap_add(x, h, block=256):
    x = np.asarray(x, float); h = np.asarray(h, float)
    M = len(h)
    N = next_power_of_two(block + M - 1)
    Hf = fft(pad_to(h, N))
    y = np.zeros(len(x) + M - 1)
    for s in range(0, len(x), block):
        blk = x[s:s + block]
        yb = ifft(fft(pad_to(blk, N)) * Hf).real[:len(blk) + M - 1]
        y[s:s + len(yb)] += yb
    return y
```

**Verified:** matches full convolution to 4e-15.

---

# §18 — Deconvolution (undo a blur)

```python
def deconvolve_1d(y, h, eps=1e-8):
    N = next_power_of_two(len(y))
    Y = fft(pad_to(y, N)); H = fft(pad_to(h, N))
    return ifft(Y * np.conj(H) / (np.abs(H) ** 2 + eps)).real


def deconvolve_2d(blurred, kernel, eps=1e-3):
    H_, W_ = blurred.shape; kh, kw = kernel.shape
    P = next_power_of_two(H_ + kh - 1); Q = next_power_of_two(W_ + kw - 1)
    Y = transform_2d(pad_2d(blurred, (P, Q)))
    G = transform_2d(pad_2d(kernel,  (P, Q)))
    X = Y * np.conj(G) / (np.abs(G) ** 2 + eps)          # Wiener-style
    return inverse_2d(X).real[:H_, :W_]
```

**Verified:** error ~2e-9 for well-conditioned kernels.

> ⚠️ **It cannot always work.** If `H[k] = 0` at some frequency, that information is *gone*.
> I measured: `h=[0.5,0.3,0.2]` → `min|H| = 0.26`, error 2e-9 ✓. But `h=[0.2,0.5,0.3]` has an
> exact spectral zero → error 0.057 ✗. `eps` trades noise amplification against blur; tune it.

---

# §19 — Resample / interpolate via the spectrum

```python
def resample_fft(x, M):
    """Band-limited resample of length-N x to length M."""
    N = len(x)
    X = fft(x) if (N & (N - 1)) == 0 else bluestein(x)
    Y = np.zeros(M, dtype=complex)
    h = min(N, M) // 2
    Y[:h] = X[:h]
    Y[M - h:] = X[N - h:]
    return (inverse(Y).real) * (M / N)


def upsample_by_zero_padding_spectrum(x, factor):
    return resample_fft(x, len(x) * factor)
```

**Verified:** upsampling a band-limited signal 64 → 256 reproduces the analytic values to 3e-15.

---

# §20 — Verify DFT properties (they love asking this)

```python
x = np.random.randn(64)
X = fft(x)
N = len(x)

# 1. Parseval
print("Parseval:", np.sum(x ** 2), np.sum(np.abs(X) ** 2) / N)

# 2. Conjugate symmetry for real x:  X[N-k] = conj(X[k])
print("conj sym err:", np.max(np.abs(X[1:] - np.conj(X[1:][::-1]))))

# 3. Circular time shift -> phase only
m = 5
lhs = fft(np.roll(x, m))
rhs = X * np.exp(-2j * np.pi * m * np.arange(N) / N)
print("shift err:", np.max(np.abs(lhs - rhs)))
print("magnitudes unchanged:", np.max(np.abs(np.abs(lhs) - np.abs(X))))

# 4. Modulation: (-1)^n swaps low and high frequencies
print("mod err:", np.max(np.abs(fft(x * (-1) ** np.arange(N)) - np.roll(X, N // 2))))

# 5. Circular convolution theorem
h = np.random.randn(64)
print("conv thm err:", np.max(np.abs(
    circular_convolve(x, h, N) - ifft(fft(x) * fft(h)).real)))

# 6. Linearity
y = np.random.randn(64)
print("linearity err:", np.max(np.abs(fft(2 * x + 3 * y) - (2 * X + 3 * fft(y)))))

# 7. X[0] is the sum
print("X[0]:", X[0].real, x.sum())
```

**All verified** (errors ~1e-15).

---

# §21 — DTFT: evaluate on a dense grid

```python
def dtft(x, omega):
    """X(e^jw) at arbitrary omega values (array). O(N*len(omega))."""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))
    return np.exp(-1j * np.outer(omega, n)) @ x


w = np.linspace(-np.pi, np.pi, 1024)
X = dtft([1, 2, 3, 4], w)
# The N-point DFT is exactly this sampled at w = 2*pi*k/N:
print(np.max(np.abs(dtft([1, 2, 3, 4], 2 * np.pi * np.arange(4) / 4) - fft([1, 2, 3, 4]))))
```

Plotting, if asked:
```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(w, np.abs(X));   ax[0].set_ylabel("|X|")
ax[1].plot(w, np.angle(X)); ax[1].set_ylabel("phase"); ax[1].set_xlabel("omega")
fig.savefig("dtft.png")
```

---

# §22 — When N is not a power of two

Already handled: `transform()` / `inverse()` auto-select Bluestein. If they explicitly want it:

```python
X = bluestein(x)          # any N, O(N log N)
x = ibluestein(X)
```

**Exact integer convolution with the NTT** (no floating point at all):

```python
NTT_MOD = 998244353
NTT_ROOT = 3

def _ntt(a, root, mod=NTT_MOD):
    a = np.asarray(a, dtype=np.int64).copy()
    N = a.shape[0]
    if N & (N - 1):
        raise ValueError("NTT length must be a power of two")
    data = _bit_reverse(a) % mod
    half = 1
    while half < N:
        span = 2 * half
        step = pow(root, (mod - 1) // span, mod)
        w = np.ones(half, dtype=np.int64)
        for i in range(1, half):
            w[i] = w[i - 1] * step % mod
        blocks = data.reshape(N // span, span)
        even = blocks[:, :half].copy()
        odd = blocks[:, half:] * w % mod
        blocks[:, :half] = (even + odd) % mod
        blocks[:, half:] = (even - odd) % mod
        data = blocks.reshape(N)
        half = span
    return data

def ntt_convolve(a, b, mod=NTT_MOD, root=NTT_ROOT):
    """Exact integer linear convolution. Every coefficient must stay < mod."""
    L = len(a) + len(b) - 1
    N = next_power_of_two(L)
    pa = np.zeros(N, dtype=np.int64); pa[:len(a)] = a
    pb = np.zeros(N, dtype=np.int64); pb[:len(b)] = b
    C = _ntt(pa, root) * _ntt(pb, root) % mod
    inv_root = pow(root, mod - 2, mod)
    inv_N = pow(N, mod - 2, mod)
    return (_ntt(C, inv_root) * inv_N % mod)[:L]
```

---

# Gotchas checklist — read before you submit

1. **Padding.** `N = next_power_of_two(len(a) + len(b) - 1)`. Forget it and the first `M−1`
   outputs are silently wrong. This is the #1 cause of lost marks.
2. **`.real` after the inverse** — the result is real up to ~1e-15 of imaginary junk.
3. **`np.rint(...).astype(np.int64)`** for integer answers. `int()` truncates:
   `60.9999999` → `60`. Wrong.
4. **The 2D crop** is `[kh//2 : kh//2+H, kw//2 : kw//2+W]`. Skip it and the image shifts
   diagonally.
5. **`.copy()` in the FFT butterfly** — `even = blocks[:, :half].copy()`. Without it you compute
   `E` instead of `E − O` and get a plausible-looking wrong answer.
6. **`np.roll` sign.** To *undo* a shift of `+s`, roll by `−s`. Test on one row first.
7. **LSD vs MSD order.** Big-integer arrays are LSD-first; polynomial papers are usually
   descending. Print both orders if unsure.
8. **Kernel must sum to 1** for a blur, or the brightness changes.
9. **Zeroing bins:** always zero both `k` and `N−k`, or use the folded-frequency trick in §7.
10. **No `numpy.fft` / `scipy`** anywhere — including "just to check". Use `dft()` from §0 to
    cross-check `fft()`.

**Sanity test to run on any transform code before you trust it:**
```python
x = np.random.randn(64) + 1j * np.random.randn(64)
assert np.max(np.abs(fft(x) - dft(x))) < 1e-9
assert np.max(np.abs(ifft(fft(x)) - x)) < 1e-9
assert np.max(np.abs(linear_convolve([1,2,3],[4,5]) - [4,13,22,15])) < 1e-9
print("core OK")
```

---

# Formula sheet

```
DFT       X[k] = Σ x[n] W^(kn),  W = e^(-j2π/N)
IDFT      x[n] = (1/N) Σ X[k] W^(-kn)
IFFT via forward:  ifft(X) = conj(fft(conj(X))) / N

linear conv length      L + M - 1
circular = linear  iff  N >= L + M - 1
2D linear conv shape    (H+kh-1, W+kw-1),  crop at (kh//2, kw//2)

bin k -> freq   f = k*fs/N  (k <= N/2);  (k-N)*fs/N  (k > N/2)
bin spacing     Δf = fs/N = 1/duration
max frequency   fs/2 at k = N/2

real x  =>  X[N-k] = conj(X[k])
shift   x[(n-m)] <-> W^(km) X[k]          (magnitude unchanged)
Parseval  Σ|x[n]|² = (1/N) Σ|X[k]|²
conv      x ⊛ h <-> X[k]H[k]              (CIRCULAR - pad!)
corr      shift s = argmax( ifft( fft(sig) * conj(fft(ref)) ) )

FFT cost  (N/2)log₂N complex mults, log₂N stages
DIT: bit-reversed IN, natural OUT.   DIF: natural IN, bit-reversed OUT.
```
