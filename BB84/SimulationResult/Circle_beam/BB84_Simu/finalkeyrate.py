import math

def compute_final_keyrate(raw_keyrate, qber, sifting_coefficient, p_estimation, kr_efficiency):
    if qber == 0:
        ab_entropy = 0
    else:
        ab_entropy = -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)
    
    ae_entropy = 1 - ab_entropy

    final_keyrate = raw_keyrate * sifting_coefficient * p_estimation * (ae_entropy - kr_efficiency * ab_entropy)
    return final_keyrate
