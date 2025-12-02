import numpy as np 

def test_spin_sym(h0):

    NF = np.shape(h0)[0]
    M = NF // 2 

    is_spin_sym = False 
    ia = [i for i in range(M)]
    ib = [i+M for i in range(M)]
    h0_a = h0[np.ix_(ia,ia)]
    h0_b = h0[np.ix_(ib,ib)]
    h0_ab = h0[np.ix_(ia,ib)]

    if np.allclose(h0_a,h0_b) and np.allclose(h0_ab,np.zeros_like(h0_ab)):
        is_spin_sym = True
    return is_spin_sym