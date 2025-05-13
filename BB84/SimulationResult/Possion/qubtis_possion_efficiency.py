import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from math import comb, log2

# Qubitの数 N_q の候補
N_q_list = [27, 28, 29]
# 実行時間（秒）
T_exec = 1.0

# 候補となる (M, k) 組
M_k_list = [(8, 2), (16, 3), (32, 4)]

# 光子数 i（縦軸に対応）
i_values = np.arange(0, 5)

# カラーマップの準備
colors = plt.cm.viridis(np.linspace(0, 1, len(i_values)))

plt.figure(figsize=(12, 8))

for N_q in N_q_list:
    for M, k in M_k_list:
        bits_per_symbol = log2(comb(M, k))
        bits_per_pulse = bits_per_symbol / k
        N_p = N_q / bits_per_symbol
        R = (k * N_p) / (T_exec * bits_per_symbol)
        n_s = bits_per_symbol / k  # 平均光子数

        # 各 i に対するポアソン分布確率
        pmf_values = [poisson.pmf(i, mu=n_s) for i in i_values]

        # プロット
        plt.plot(i_values, pmf_values, marker='o',
                 label=f'N_q={N_q}, M={M}, k={k}, n_s={n_s:.2f}')

plt.xlabel('Number of detected photons (i)')
plt.ylabel('Poisson Probability P(i; μ = n_s)')
plt.title('Poisson Distribution vs. Average Photon Number $n_s$ from MPPM')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()