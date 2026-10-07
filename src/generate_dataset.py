"""
Synthetic wireless dataset generator for Automatic Modulation Classification (AMC).

The received baseband signal model is

    y[n] = x[n] * h[n] + w[n]

where x[n] is the transmitted symbol stream (pulse shaped), h[n] the channel
(here: flat, h = 1) and w[n] complex additive white Gaussian noise (AWGN)
whose power is set so the signal is observed at a target SNR.

For every signal block, higher-order statistical features (normalized
cumulants C20, C40, C41, C42) plus simple spectral/energy features and the
true SNR are extracted. These features are classical AMC hand-crafted
features (see e.g. Swami & Sadler).
"""

import numpy as np
import pandas as pd

CLASSES = ["BPSK", "QPSK", "16QAM"]
FEATURES = ["c20_abs", "c40", "c41", "c42", "energy", "spectral_entropy", "snr"]


# ----------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------
def random_symbols(modulation: str, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_symbols` random unit-average-energy symbols."""
    if modulation == "BPSK":
        return rng.choice([-1.0, 1.0], size=n_symbols).astype(complex)
    if modulation == "QPSK":
        i = rng.choice([-1.0, 1.0], size=n_symbols)
        q = rng.choice([-1.0, 1.0], size=n_symbols)
        return (i + 1j * q) / np.sqrt(2.0)
    if modulation == "16QAM":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        i = rng.choice(levels, size=n_symbols)
        q = rng.choice(levels, size=n_symbols)
        return i + 1j * q
    raise ValueError(f"Unknown modulation {modulation}")


def rrc_pulse(sps: int, span: int, beta: float = 0.35) -> np.ndarray:
    """Root-raised-cosine FIR filter (normalized to unit energy)."""
    n = np.arange(-span * sps, span * sps + 1)
    h = np.zeros_like(n, dtype=float)
    for k, t in enumerate(n):
        t = t / sps  # time in symbol periods
        if t == 0.0:
            h[k] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(t) - 1.0 / (4.0 * beta)) < 1e-8:
            h[k] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(np.pi * t * (1.0 + beta))
            den = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
            h[k] = num / den
    return h / np.sqrt(np.sum(h**2))


def generate_block(modulation: str, n_symbols: int, sps: int,
                   snr_db: float, rng: np.random.Generator,
                   beta: float = 0.35) -> np.ndarray:
    """One received signal block at the requested SNR (flat channel)."""
    tx = random_symbols(modulation, n_symbols, rng)
    pulse = rrc_pulse(sps, span=6, beta=beta)
    up = np.zeros(n_symbols * sps, dtype=complex)
    up[::sps] = tx
    x = np.convolve(up, pulse, mode="full")

    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    )
    return x + noise


# ----------------------------------------------------------------------
# Feature extraction (normalized higher-order cumulants + extras)
# ----------------------------------------------------------------------
def cumulant_features(y: np.ndarray) -> dict:
    """
    Normalized cumulants of a complex signal y (Swami & Sadler notation).

    C21 = E[|y|^2]                    (average power, used for normalization)
    C20 = E[y^2]   / C21
    C40 = (E[y^4]           - 3 E[y^2]^2)                    / C21^2
    C41 = (E[y^3 conj(y)]   - 3 E[y^2] E[|y|^2])              / C21^2
    C42 = (E[|y|^4]         - |E[y^2]|^2 - 2 E[|y|^2]^2)      / C21^2
    """
    c21 = np.mean(np.abs(y) ** 2)
    m20 = np.mean(y ** 2)
    m40 = np.mean(y ** 4)
    m41 = np.mean((y ** 3) * np.conj(y))
    m42 = np.mean(np.abs(y) ** 4)

    c20 = m20 / c21
    c40 = (m40 - 3.0 * m20 ** 2) / c21 ** 2
    c41 = (m41 - 3.0 * m20 * c21) / c21 ** 2
    c42 = (m42 - np.abs(m20) ** 2 - 2.0 * c21 ** 2) / c21 ** 2
    return {"c20_abs": np.abs(c20), "c40": c40.real,
            "c41": np.abs(c41), "c42": c42}


def spectral_entropy(y: np.ndarray) -> float:
    """Shannon entropy of the normalized PSD (a simple cyclo/spectral cue)."""
    psd = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    p = psd / psd.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_features(y: np.ndarray, snr_db: float) -> dict:
    feats = cumulant_features(y)
    feats["energy"] = float(np.mean(np.abs(y) ** 2))
    feats["spectral_entropy"] = spectral_entropy(y)
    feats["snr"] = snr_db
    return feats


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------
def build_dataset(snr_list, samples_per_class_snr=200, n_symbols=512,
                  sps=4, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for snr in snr_list:
        for mod in CLASSES:
            for i in range(samples_per_class_snr):
                y = generate_block(mod, n_symbols, sps, snr,
                                   rng=np.random.default_rng(seed + i + abs(int(round(1000 * snr)))))
                feats = extract_features(y, snr)
                feats["modulation"] = mod
                rows.append(feats)
            print(f"done  SNR={snr:>3} dB  {mod} "
                  f"({samples_per_class_snr} samples)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AMC dataset")
    parser.add_argument("--out", default="data/dataset.csv")
    parser.add_argument("--snr-min", type=float, default=-10)
    parser.add_argument("--snr-max", type=float, default=20)
    parser.add_argument("--snr-step", type=float, default=2)
    parser.add_argument("--samples", type=int, default=200,
                        help="samples per (class, SNR)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snrs = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
    df = build_dataset(snrs, samples_per_class_snr=args.samples, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {df.shape[0]} rows -> {args.out}")
