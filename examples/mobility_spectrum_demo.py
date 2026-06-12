"""Run mobility spectrum analysis on a synthetic conductivity spectrum."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.msa.kernel import build_msa_kernel
from transport_toolkit.msa.mobility_spectrum import run_msa
from transport_toolkit.msa.peaks import find_msa_peaks


field = np.linspace(-2, 2, 31)
mobility = np.linspace(-3, 3, 121)
mobility = mobility[mobility != 0]

_, k_xx, k_xy = build_msa_kernel(field, mobility)
truth = 2.0 * np.exp(-((mobility + 1.0) / 0.25) ** 2) + 1.5 * np.exp(-((mobility - 1.2) / 0.35) ** 2)
sigma_xx = k_xx @ truth
sigma_xy = k_xy @ truth

result = run_msa(field, sigma_xx, sigma_xy, mobility=mobility, alpha=1e-4)
peaks = find_msa_peaks(result.mobility, result.spectrum)

print(result.to_dict())
print([peak.to_dict() for peak in peaks[:4]])
