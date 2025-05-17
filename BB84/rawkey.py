import matplotlib.pyplot as plt
import os

# データの定義
num_qubits = [50, 100, 150, 200, 250, 300, 350, 400]
raw_key_rate = [4185.15, 4715.18, 4866.37, 4963.07, 4977.82, 4940.16, 4838.11, 4585.40]

# プロット
plt.figure(figsize=(10, 6))
plt.plot(num_qubits, raw_key_rate, marker='o', linestyle='-')

# ラベルとタイトル
plt.xlabel('Number of Qubits', fontsize=20)
plt.ylabel('Raw Key Rate (Qubits/sec)', fontsize=20)
plt.title('Raw Key Rate vs Number of Qubits', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "raw_key_rate_mac.png")
plt.savefig(output_path)
print(f"✅ Saved as: {output_path}")

plt.show()
