import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 平均光子数の候補（横軸）
ns_values = np.arange(0.1, 1.1, 0.1)
# 光子数 i（カテゴリごとの色分け）
i_values = np.arange(0, 5)

# 棒の幅と位置調整
bar_width = 0.12
x = np.arange(len(ns_values))  # μ の個数に対応する x 軸の位置

# カラーマップで色を設定
colors = plt.cm.tab10(np.linspace(0, 1, len(i_values)))

# グラフの作成
plt.figure(figsize=(10, 6))

for idx, i in enumerate(i_values):
    pmf_values = [poisson.pmf(i, mu=ns) for ns in ns_values]
    offset = (idx - len(i_values)/2) * bar_width + bar_width/2
    plt.bar(x + offset, pmf_values, width=bar_width, color=colors[idx], edgecolor='black', label=f'i = {i}')

# 軸とラベル設定
plt.xticks(x, [f'{ns:.1f}' for ns in ns_values])
plt.xlabel('μ (Average number of photons)')
plt.ylabel('P(i photons)')
plt.title('Poisson Probability vs. Average Photon Number')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Photon count (i)')
plt.tight_layout()
plt.show()
