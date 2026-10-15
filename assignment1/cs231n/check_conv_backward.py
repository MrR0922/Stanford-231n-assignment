"""Numerical gradient check for the implementation supplied in chat."""
import numpy as np
from gradient_check import eval_numerical_gradient_array


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    part = x_pad[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    dw[f] += part * dout[n, f, i, j]
                    dx_pad[n, :, h_start:h_start + HH, w_start:w_start + WW] += w[f] * dout[n, f, i, j]
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db


def forward_reference(x, w, b, param):
    p, s = param["pad"], param["stride"]
    padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, w.shape[2:], axis=(2, 3))
    windows = windows[:, :, ::s, ::s, :, :]
    return np.einsum("ncijuv,fcuv->nfij", windows, w) + b[None, :, None, None]


def main():
    rng = np.random.RandomState(231)
    cases = [
        ("standard", (4, 3, 5, 5), (2, 3, 3, 3), 1, 1),
        ("no_padding", (2, 2, 5, 6), (3, 2, 3, 2), 0, 1),
        ("stride_two", (2, 2, 5, 7), (3, 2, 3, 3), 1, 2),
        ("rectangular", (2, 2, 4, 6), (2, 2, 2, 4), 1, 2),
        ("pointwise", (2, 3, 3, 4), (2, 3, 1, 1), 0, 1),
    ]
    for name, xshape, wshape, pad, stride in cases:
        x = rng.randn(*xshape)
        w = rng.randn(*wshape)
        b = rng.randn(wshape[0])
        param = {"pad": pad, "stride": stride}
        dout = rng.randn(*forward_reference(x, w, b, param).shape)
        original = [a.copy() for a in (x, w, b, dout)]
        analytic = conv_backward_naive(dout, (x, w, b, param))
        numeric = [
            eval_numerical_gradient_array(lambda v: forward_reference(v, w, b, param), x, dout),
            eval_numerical_gradient_array(lambda v: forward_reference(x, v, b, param), w, dout),
            eval_numerical_gradient_array(lambda v: forward_reference(x, w, v, param), b, dout),
        ]
        print(f"{name}: pad={pad}, stride={stride}")
        for label, actual, expected in zip(("dx", "dw", "db"), analytic, numeric):
            assert actual.shape == expected.shape
            rel = np.max(np.abs(actual - expected) / np.maximum(1e-8, np.abs(actual) + np.abs(expected)))
            absolute = np.max(np.abs(actual - expected))
            print(f"  {label}: max_relative_error={rel:.3e}, max_absolute_error={absolute:.3e}")
            assert rel < 1e-6, (name, label, rel)
        for current, saved in zip((x, w, b, dout), original):
            assert np.array_equal(current, saved), "Input was mutated"
    print("PASS: all five cases; relative errors < 1e-6; shapes correct; inputs unchanged.")


if __name__ == "__main__":
    main()
