import math




def compute_secret_keyrate(qber, param_q, sifting_coefficient, p_estimation, kr_efficiency):
    if qber == 0:
        ab_entropy = 0
    else:
        ab_entropy = -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)
    
    ae_entropy = 1 - ab_entropy

    term_1  = ae_entropy - kr_efficiency * ab_entropy

    if (term_1 <=0):
        final_keyrate = 0
    else:
        final_keyrate = param_q * sifting_coefficient * p_estimation * term_1
    # print(ae_entropy - kr_efficiency * ab_entropy)
    return final_keyrate
