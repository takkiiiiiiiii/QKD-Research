# import math 


# sifting_coefficiant = 0.5
# p_estimation = 0.9
# kr_efficiency = 1.22
# qber = 0
# raw_keyrate = 368.0363931165903

# if qber == 0:
#     ab_entropy = 0
# else:
#     ab_entropy = -qber*math.log2(qber)-(1-qber)*math.log2(1-qber)

# ae_entropy = 1-ab_entropy

# # final_keyrate = raw_keyrate*sifting_coefficiant*p_estimation*(ae_entropy-kr_efficiency*ab_entropy)

# final_keyrate = raw_keyrate*sifting_coefficiant*p_estimation*(ae_entropy)
# print(final_keyrate)


import math

def compute_final_keyrate(raw_keyrate, qber, sifting_coefficient, p_estimation, kr_efficiency):
    if qber == 0:
        ab_entropy = 0
    else:
        ab_entropy = -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)
    
    ae_entropy = 1 - ab_entropy

    final_keyrate = raw_keyrate * sifting_coefficient * p_estimation * (ae_entropy - kr_efficiency * ab_entropy)
    return final_keyrate
