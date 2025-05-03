import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os, sys
import numpy as np
from scipy.integrate import quad
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)

# from fading import rytov_variance, cn2_profile

# Constants
len_wave = 0.85e-6  # [m] 0.85 μm
H_OGS = 122           # Ground station height [m]

v_wind = 21         # Wind speed [m/s]

# Hufnagel-Valley Cn2 profile
def hufnagel_valley_cn2(h, Cn2_0):
    term1 = 0.00594 * (v_wind / 27)**2 * (10**-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3

# Rytov variance computation
def rytov_variance(len_wave, theta_zen_rad, H_OGS, H_atm, Cn2_profile):
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(theta_zen_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - H_OGS)**(5/6)

    integral, _ = quad(integrand, H_OGS, H_atm, limit=100, epsabs=1e-9, epsrel=1e-9)
    sigma_R_squared = 2.25 * k**(7/6) * sec_zenith**(11/6) * integral
    return sigma_R_squared

# Simulation over varying Cn2(0) and zenith angles
Cn2_0_values = np.logspace(-17, -13, 100)
theta_zen_degrees = np.arange(30, 71, 10)
results = {}

for theta_deg in theta_zen_degrees:
    theta_rad = np.radians(theta_deg)
    # H_atm = 20000 * np.cos(theta_rad)  # Effective atmosphere thickness
    H_atm = 20000
    rytov_vals = []

    for Cn2_0 in Cn2_0_values:
        profile = lambda h: hufnagel_valley_cn2(h, Cn2_0)
        sigma_R2 = rytov_variance(len_wave, theta_rad, H_OGS, H_atm, profile)
        rytov_vals.append(sigma_R2)

    results[theta_deg] = rytov_vals

# Plotting
plt.figure(figsize=(10, 6))
for theta_deg in theta_zen_degrees:
    plt.plot(Cn2_0_values, results[theta_deg], label=f'{theta_deg}°')

plt.xscale('log')
plt.yscale('linear')
plt.xlabel(r'$C_n^2(0)$ [m$^{-2/3}$]', fontsize=14)
plt.ylabel(r'Rytov Variance $\sigma_R^2$', fontsize=14)
plt.title('Rytov variance versus ground level turbulence for different Zenith Angles', fontsize=16)
plt.grid(True)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend(title='Zenith Angle', fontsize=12)
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), f'Rytov_variance_zenith.png')
plt.savefig(output_path)
print(f"✅ Saved as: {output_path}")
plt.show()
