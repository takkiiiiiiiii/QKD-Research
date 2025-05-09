import numpy as np


def Hamming_kr_one_block(ka, kb):
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])

    syndrome_A = np.dot(ka, np.transpose(H)) % 2
    syndrome_B = np.dot(kb, np.transpose(H)) % 2

    syn_diff = (syndrome_A + syndrome_B) % 2
    err_pos = int(syn_diff[0] + syn_diff[1] * 2 + syn_diff[2] * 4)

    reconciled_kb = np.copy(kb)
    if (err_pos != 0):
        reconciled_kb[err_pos-1] = 1 - reconciled_kb[err_pos-1]

    return ((ka == reconciled_kb).all())


def key_reconciliation_Hamming(ka, kb):
    """Key reconciliation using (4, 7) Hamming code

    Args:
        ka (numpy array): Alice's sifted key
        kb (numpy array): Bob's sifted key
    """
    ka_array = np.array([int(char) for char in ka])
    kb_array = np.array([int(char) for char in kb])

    length_A = len(ka_array)
    length_B = len(kb_array)
    print(length_A)

    reconciled_key = np.array([])

    assert ((length_A % 7) == 0)
    assert (length_A == length_B)

    # Number of blocks
    n_blocks = int(length_A/7)

    for idx in range(n_blocks):
        block_A = ka_array[(idx)*7:(idx+1)*7]
        block_B = kb_array[(idx)*7:(idx+1)*7]
        if (Hamming_kr_one_block(block_A, block_B)):
            reconciled_key = np.append(reconciled_key, block_A)
    return reconciled_key
