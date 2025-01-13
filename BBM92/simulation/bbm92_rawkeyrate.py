import numpy as np
import matplotlib.pyplot as plt

qbit = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28])

raw_keyrate = np.array([67.60077722218934,
                        126.56278759507904,
                        188.22375349636562,
                        254.98181870693278,
                        309.90519136588443,
                        355.1669282615894,
                        398.2479978882338,
                        453.1592758108533,
                        507.7398852265206,
                        549.3431244087261,
                        577.3619453251443,
                        618.1921150008482,
                        655.6529358655627,
                        693.0715277435713,
                        ])



runtime = np.array([0.01669027018547058,
                    0.017408823490142823,
                    0.017084189653396607,
                    0.01655804920196533,
                    0.017227375268936158,
                    0.018229294300079347,
                    0.018587010383605957,
                    0.018447007179260254,
                    0.018343766927719118,
                    0.018952183246612547,
                    0.01981734085083008,
                    0.020299697875976564,
                    0.02051921367645264,
                    0.020881993293762207,
                    ])

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Generated Qubits from IBM Platform per execution', fontsize=22)
ax1.set_ylabel('Raw Key Rate (bps)', color=color, fontsize=22)
ax1.plot(qbit, raw_keyrate, color=color, marker='o', label='Sifted Key Rate (bps)', markersize=10)
ax1.tick_params(axis='y', labelcolor=color, labelsize=22)
ax1.tick_params(axis='x', labelsize=20)
ax1.set_ylim(50, 700)
ax1.set_xlim(0, 28)
ax1.set_xticks(np.arange(0, 29, 2))
ax1.set_yticks(np.arange(0, 700, 50))
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Execution time(s)', color=color, fontsize=22)
ax2.plot(qbit, runtime, color=color, marker='s', label='Runtime (s)', markersize=10)
ax2.tick_params(axis='y', labelcolor=color, labelsize=22)
ax2.set_ylim(0, 0.021)
ax2.set_yticks(np.arange(0, 0.021, 0.01))
ax2.grid(alpha=0.3)

# fig.suptitle('Generated Qubits from IBM Platform per Execution vs. Sifted Key Rate and Runtime')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('qlength_vs_raw_sifted_keyrate.png', format='png', bbox_inches="tight", dpi=300)

plt.show()