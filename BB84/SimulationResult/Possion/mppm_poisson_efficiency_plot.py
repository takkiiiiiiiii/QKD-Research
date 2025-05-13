import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from math import comb, log2

# 候補パラメータ
M_k_list = [(8, 2), (16, 3), (32, 4)]  # MPPMの (M, k) 組
ns_values = np.linspace(0.1, 1.0, 10)  # 平均光子数 μ の値（0.1〜1.0）

plt.figure(figsize=(10, 6))

for M, k in M_k_list:
    bits_per_pulse = log2(comb(M, k)) / k
    detect_probs = []
    for ns in ns_values:
        # 光子が1個以上届く確率 = 1 - P(0)
        p_detect = 1 - poisson.pmf(0, mu=ns)
        detect_probs.append(bits_per_pulse * p_detect)  # 検出成功時の実効情報量（bits/pulse）

    label = f'(M={M}, k={k}) - {bits_per_pulse:.2f} bits/pulse'
    plt.plot(ns_values, detect_probs, marker='o', label=label)

plt.xlabel('Average photons per pulse (μ)')
plt.ylabel('Effective bits per pulse (with detection)')
plt.title('MPPM Efficiency vs. Photon Detection Probability')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

