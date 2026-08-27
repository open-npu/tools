"""MSE-clip and ADMM weight PTQ helpers.

Hardware stays symmetric zp=0 (INT8/INT16). These methods only change how
activation clips and per-channel weight scales/integers are chosen.

- MSE: search clip T that minimizes ||x - quant(x, T)||^2
- ADMM: project each weight channel onto the quantized lattice with a
  quadratic consensus step, then encode the consensus as (w_q, scale)
"""

import numpy as np


def qmax_for(bits):
    return (1 << (bits - 1)) - 1


def fake_quant(x, scale, bits):
    qmax = qmax_for(bits)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale


def mse_optimal_scale(values, bits=8, n_grid=32, lo=0.35):
    """Scale s = T/qmax minimizing reconstruction MSE on `values`."""
    x = np.asarray(values, dtype=np.float64).ravel()
    qmax = qmax_for(bits)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs < 1e-12:
        return 1e-10
    best_s = max_abs / qmax
    best_mse = np.mean((x - fake_quant(x, best_s, bits)) ** 2)
    for t in np.linspace(lo * max_abs, max_abs, n_grid):
        s = t / qmax
        if s < 1e-12:
            continue
        mse = float(np.mean((x - fake_quant(x, s, bits)) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_s = s
    return best_s


def mse_clip_range(samples, bits=8, is_relu=False):
    """Return (vmin, vmax) for symmetric (or ReLU) fake-quant with MSE clip."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        return (0.0, 1.0) if is_relu else (-1.0, 1.0)
    if is_relu:
        x = np.maximum(x, 0.0)
    s = mse_optimal_scale(x, bits=bits)
    t = s * qmax_for(bits)
    if is_relu:
        return 0.0, float(t)
    return -float(t), float(t)


def admm_quantize_channel(w0, bits=8, iters=20, rho=1.5):
    """ADMM projection of one weight channel onto the symmetric quantized set.

    Returns integer codes and the scale used to dequant them.
    """
    w0 = np.asarray(w0, dtype=np.float64).ravel()
    qmax = qmax_for(bits)
    if w0.size == 0 or float(np.max(np.abs(w0))) < 1e-12:
        return np.zeros_like(w0, dtype=np.int32), 1e-10

    s = mse_optimal_scale(w0, bits=bits, n_grid=24)
    z = fake_quant(w0, s, bits)
    u = np.zeros_like(w0)
    for _ in range(iters):
        x = (w0 + rho * (z - u)) / (1.0 + rho)
        s = mse_optimal_scale(x, bits=bits, n_grid=16, lo=0.4)
        z = fake_quant(x + u, s, bits)
        u = u + x - z

    # Encode the ADMM consensus z as integer weights + one scale.
    s = max(float(np.max(np.abs(z))), 1e-12) / qmax
    w_q = np.clip(np.round(z / s), -qmax, qmax).astype(np.int32)
    return w_q, s


def admm_quantize_weight_perchannel(weight, out_axis=0, bits=8, iters=20, rho=1.5):
    """Per-channel ADMM weight quant. Same contract as quantize_weight_perchannel."""
    if out_axis != 0:
        raise ValueError("Unsupported out_axis")
    qmax = qmax_for(bits)
    out_c = weight.shape[0]
    dtype = np.int8 if bits == 8 else np.int16
    weight_q = np.zeros_like(weight, dtype=dtype)
    scale_w = np.zeros(out_c, dtype=np.float64)
    for c in range(out_c):
        w_q, s = admm_quantize_channel(weight[c], bits=bits, iters=iters, rho=rho)
        scale_w[c] = s
        weight_q[c] = w_q.reshape(weight[c].shape).astype(dtype)
        # Guard empty / overflow
        if scale_w[c] < 1e-12:
            scale_w[c] = 1e-10
        np.clip(weight_q[c], -qmax, qmax, out=weight_q[c])
    return weight_q, scale_w


def reservoir_chunks(sample_chunks, rng, limit=16384):
    """Concatenate per-image chunks and downsample to `limit` values."""
    out = {}
    for name, chunks in sample_chunks.items():
        cat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if cat.size > limit:
            out[name] = rng.choice(cat, limit, replace=False)
        else:
            out[name] = cat
    return out
