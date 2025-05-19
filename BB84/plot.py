import matplotlib.pyplot as plt

# データ
num_qubits_list = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
raw_keyrates = [394.76, 3873.46, 4639.69, 4654.06, 4891.08, 5077.23, 5023.35, 4948.22,
                4840.17, 4700.80, 4696.11, 4608.96, 4564.92, 4443.28, 4317.26, 4205.05,
                4146.63, 3863.44, 3696.45, 3773.82]
runtimes = [0.00143, 0.00635, 0.01069, 0.01619, 0.02058, 0.02489, 0.02985, 0.03526,
            0.04150, 0.04778, 0.05343, 0.06008, 0.06587, 0.07338, 0.08164, 0.08955,
            0.09717, 0.10974, 0.12146, 0.12610]

# 最大値を探す
max_keyrate = max(raw_keyrates)
max_index = raw_keyrates.index(max_keyrate)
max_qubit = num_qubits_list[max_index]

# プロット
fig, ax1 = plt.subplots(figsize=(12, 7))

# 左軸: Raw Key Rate
color1 = 'tab:blue'
ax1.set_xlabel("Number of Qubits", fontsize=16)
ax1.set_ylabel("Raw Key Rate (Qubit/sec)", color=color1, fontsize=16)
ax1.plot(num_qubits_list, raw_keyrates, marker='o', linestyle='-', color=color1, label="Raw Key Rate")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# 最大点にマーカーと注釈を追加
ax1.plot(max_qubit, max_keyrate, 'ro', label='Max Raw Key Rate')
ax1.annotate(f"Max: {max_keyrate:.2f} Q/s\nat {max_qubit} qubits",
             xy=(max_qubit, max_keyrate),
             xytext=(max_qubit + 50, max_keyrate - 500),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=14, color='red')

# 右軸: Runtime
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Runtime (sec)", color=color2, fontsize=16)
ax2.plot(num_qubits_list, runtimes, marker='s', linestyle='--', color=color2, label="Runtime")
ax2.tick_params(axis='y', labelcolor=color2)
ax2.tick_params(axis='y', labelsize=14)

# タイトルなど
fig.suptitle("Raw Key Rate & Runtime vs Number of Qubits", fontsize=20)
fig.tight_layout()
plt.show()
