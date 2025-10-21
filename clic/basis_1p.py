import numpy as np

def transform_integrals_interleaved_to_alphafirst(h0_int, U_int, M):
    """
    Transforms integrals from spin-interleaved to AlphaFirst ordering.
    """
    K = 2 * M
    af_map = np.zeros(K, dtype=int)
    for i in range(M):
        af_map[i] = 2 * i
        af_map[i + M] = 2 * i + 1

    h0_af = np.zeros_like(h0_int)
    for p_af in range(K):
        for q_af in range(K):
            h0_af[p_af, q_af] = h0_int[af_map[p_af], af_map[q_af]]
            
    U_af = np.zeros_like(U_int)
    for p_af in range(K):
        for q_af in range(K):
            for r_af in range(K):
                for s_af in range(K):
                    p_int, q_int = af_map[p_af], af_map[q_af]
                    r_int, s_int = af_map[r_af], af_map[s_af]
                    U_af[p_af, q_af, r_af, s_af] = U_int[p_int, q_int, r_int, s_int]
                    
    return h0_af, U_af