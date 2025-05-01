from scipy.integrate import quad
import math
import numpy as np
from fading import fading_loss
from qber import qber_loss



def qner_new_infinite():
    def integrand(gamma):
        return fading_loss(gamma) * qber_loss(gamma)
    result, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result


def main():
    qber = qner_new_infinite()
    print(f'QBER: {qber}')

if __name__ == "__main__":
    main()


