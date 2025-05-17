import matplotlib.pyplot as plt
import os

num_qubits = [50, 100, 150, 200, 250, 300, 350, 400]
raw_key_rate = [4100.70, 4784.82, 5127.48, 5267.55, 5307.12, 5067.82, 4901.75, 4860.84]


plt.figure(figsize=(10, 6))
plt.plot(num_qubits, raw_key_rate, marker='o', linestyle='-')
plt.axvline(x=250, color='red', linestyle='--', label='Peak at 250 Qubits')


plt.xlabel('Number of Qubits', fontsize=20)
plt.ylabel('Raw Key Rate (Qubits/sec)', fontsize=20)
plt.title('Raw Key Rate vs Number of Qubits', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "raw_key_rate_linux.png")
plt.savefig(output_path)
print(f"✅ Saved as: {output_path}")

plt.show()
