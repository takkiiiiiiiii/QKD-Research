from scipy.integrate import quad
import math
import numpy as np
from fading import fading_loss
from qber import qber_loss



a = 0.75      # Aperture of radius (Receiver radis in meters)
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.8]
mag_w2 = [0.1, 0.9, 1.7]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]

def qner_new_infinite():
    def integrand(gamma):
        return fading_loss(gamma) * qber_loss(gamma)
    result, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result


def main():
    qber = qner_new_infinite()
    print(f'QBER: {qber}')

if __name__ == "__main__":
    main()


