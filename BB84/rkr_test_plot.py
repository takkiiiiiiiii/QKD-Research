import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

# データ
num_qubits_list = [
    1, 50, 100, 150, 200, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
    1000, 1050, 1100,
    1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600,
    1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100,
    2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550,
    2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950
]

raw_keyrates = [
    394.76, 3873.46, 4639.69, 4654.06, 4891.08, 5077.23, 5023.35, 4948.22, 4840.17, 4700.80,
    4696.11, 4608.96, 4564.92, 4443.28, 4317.26, 4205.05, 4146.63, 3863.44, 3696.45, 3773.82,
    3653.62, 3486.02, 3325.44,
    3288.97, 3221.23, 3099.94, 2996.42, 2884.58, 2845.56, 2730.89, 2592.49, 2497.62, 2417.20,
    2290.91, 2174.82, 2082.63, 1986.22, 1943.35, 1613.41, 1548.22, 1654.25, 1582.38, 1531.99,
    1471.61, 1410.52, 1324.71, 1290.58, 1249.70, 1293.27, 1249.18, 1075.00, 1068.17,
    1043.35, 1015.27, 987.94, 969.21, 935.09, 909.84, 812.44, 785.60
]

runtimes = [
    0.00143, 0.00635, 0.01069, 0.01619, 0.02058, 0.02489, 0.02985, 0.03526, 0.04150, 0.04778,
    0.05343, 0.06008, 0.06587, 0.07338, 0.08164, 0.08955, 0.09717, 0.10974, 0.12146, 0.12610,
    0.13677, 0.14980, 0.16549,
    0.17564, 0.18560, 0.20195, 0.21710, 0.23464, 0.24523, 0.26654, 0.28813, 0.31019, 0.33206,
    0.36024, 0.39220, 0.41945, 0.45469, 0.47574, 0.59118, 0.62943, 0.60616, 0.64911, 0.68591,
    0.72907, 0.77622, 0.84867, 0.89120, 0.93946, 0.92744, 0.98032, 1.16255, 1.19258,
    1.24516, 1.30361, 1.36753, 1.41974, 1.49533, 1.56988, 1.77907, 1.87501
]
runtimes_ms = [t * 1000 for t in runtimes]


# 最大 Raw Key Rate を検出
max_keyrate = max(raw_keyrates)
max_index = raw_keyrates.index(max_keyrate)
max_qubit = num_qubits_list[max_index]

# グラフ描画
fig, ax1 = plt.subplots(figsize=(14, 8))

# 左軸: Raw Key Rate
color1 = 'tab:blue'
ax1.set_xlabel("Generated Qubits from IQX Simulator per execution", fontsize=30)
ax1.set_ylabel("Raw Key Rate (Qubit/sec)", color=color1, fontsize=30)

ax1.plot(num_qubits_list, raw_keyrates, marker='o', linestyle='-', color=color1, linewidth=4.5)

ax1.tick_params(axis='y', labelcolor=color1, labelsize=29)
ax1.tick_params(axis='x', labelsize=30)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xticks(range(0, 2001, 250))

ax1.set_xlim(0, 2000)

# ax1.axvline(
#     x=max_qubit,
#     color='purple',
#     linestyle='--',
#     linewidth=3.5,
# )

ax1.annotate(
    'Peak at 250 Qubits',
    xy=(max_qubit, max_keyrate),                  
    xytext=(500, 5000),    # 注釈テキストの位置
    arrowprops=dict(arrowstyle='->', color='black', lw=3.5),
    fontsize=25,
    color='black'
)

# 小さな横型楕円を追加（matplotlib.patches.Ellipseを使用）
# ellipse = patches.Ellipse((1500, 2500), width=200, height=200, edgecolor='black', facecolor='none', lw=1)
# ax1.add_patch(ellipse)
# arrow_target_x = 1500 + 100 * np.cos(np.pi / 4)  # width方向の半径 * cos(45°)
# arrow_target_y = 2500 + 100 * np.sin(np.pi / 4)  # height方向の半径 * sin(45°)

# 矢印とラベルの追加
# ax1.annotate(
#     'hybrid',
#     xy=(arrow_target_x, arrow_target_y),
#     xytext=(1650, 2850),  # ラベル位置
#     arrowprops=dict(arrowstyle='<-', color='black', lw=2),
#     fontsize=22,
#     color='black'
# )

# 凡例表示
# ax1.legend(fontsize=25)

# 右軸: Runtime
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Execution Time (ms)", color=color2, fontsize=30)
ax2.plot(num_qubits_list, runtimes_ms, marker='s', linestyle='--', color=color2, label='Execution Time', linewidth=4.0)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=30)
# y軸範囲と目盛り（msで指定）
ax2.set_ylim(0, 650)  # 0～150ms
yticks = np.arange(100, 651, 100)  # 30ms刻み
ax2.set_yticks(yticks)
# タイトル
# fig.suptitle("Raw Key Rate & Runtime vs Number of Qubits", fontsize=22)
fig.tight_layout()
# output_path = os.path.join(os.path.dirname(__file__), "raw_key_rate_runtime.pdf")
# plt.savefig(output_path, format='pdf')
# print(f"✅ Saved as: {output_path}")
plt.savefig('raw_key_rate_runtime.pdf', format='pdf', bbox_inches="tight")
