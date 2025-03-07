import numpy as np
import matplotlib.pyplot as plt

# データの定義
mu_values = np.array([0.1, 0.5, 0.7, 1.0])
transmittance_levels = [0.1, 0.5, 0.9]
received_photons = {
    0.1: [38672.48143059954, 7734.496286119909, 5524.640204371365, 3867.2481430599546],
    0.5: [7734.496286119909, 1546.899257223982, 1104.928040874273, 773.449628611991],
    0.9: [4296.942381177727, 859.3884762355456, 613.8489115968182, 429.6942381177728]
}

# バーの幅を調整
width = 0.05  # 以前より細めに設定

plt.figure(figsize=(10, 6))

# Transmittanceごとにバーを並べる
for i, trans in enumerate(transmittance_levels):
    plt.bar(mu_values + (i - 1) * width, received_photons[trans], width, label=f"Transmittance {trans}", align="center")

# 軸ラベルとタイトル
plt.xlabel("μ (Average Photon Number)")
plt.ylabel("Average No. of Received Photons (photon/second)")
plt.title("Average Received Photons vs. μ")
plt.yscale("log")  # 受信光子数が大きく異なるため、対数スケールを使用
plt.xticks(mu_values)  # X軸の目盛りをμの値に設定

# 縦軸の範囲をデータに合わせて調整
y_min = min([min(received_photons[trans]) for trans in transmittance_levels])
y_max = max([max(received_photons[trans]) for trans in transmittance_levels])
plt.ylim([y_min/2, y_max*1.5])  # データの最小値と最大値に余裕をもたせる

plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)

# グラフの表示
plt.show()
