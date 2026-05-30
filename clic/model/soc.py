import numpy as np

def get_soc_ham(l,soc_lambda) : 
    s = 0.5
    # Lambda parameter (coupling strength)

    # 2. Generate the Basis States |ml, ms>
    # Total dimension = (2l+1) * (2s+1) = 7 * 2 = 14
    basis = []
    for ms in [0.5, -0.5]:
        for ml in range(-l, l + 1):
            basis.append({'ml': ml, 'ms': ms})
            
    dim = len(basis)
    print(f"Basis dimension: {dim}x{dim}\n")

    # 3. Define Helper Functions for Angular Momentum Operators
    
    # Ladder operator coefficient: sqrt(j(j+1) - m(m+1))
    def c_plus(j, m):
        return np.sqrt(j*(j+1) - m*(m+1))
    
    def c_minus(j, m):
        return np.sqrt(j*(j+1) - m*(m-1))

    # 4. Construct the Hamiltonian Matrix
    # H = lambda * (LzSz + 0.5 * (L+S- + L-S+))
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):     # Bra <i|
        for j in range(dim): # Ket |j>
            
            bra = basis[i]
            ket = basis[j]
            
            # --- Term 1: Lz * Sz ---
            # Only non-zero if bra == ket (diagonal)
            val_LzSz = 0
            if i == j:
                val_LzSz = ket['ml'] * ket['ms']
            
            # --- Term 2: 0.5 * L+ * S- ---
            # Raises ml by 1, lowers ms by 1
            val_LpSm = 0
            if (bra['ml'] == ket['ml'] + 1) and (bra['ms'] == ket['ms'] - 1):
                val_LpSm = 0.5 * c_plus(l, ket['ml']) * c_minus(s, ket['ms'])
                
            # --- Term 3: 0.5 * L- * S+ ---
            # Lowers ml by 1, raises ms by 1
            val_LmSp = 0
            if (bra['ml'] == ket['ml'] - 1) and (bra['ms'] == ket['ms'] + 1):
                val_LmSp = 0.5 * c_minus(l, ket['ml']) * c_plus(s, ket['ms'])
                
            # Sum them up
            H[i, j] = soc_lambda * (val_LzSz + val_LpSm + val_LmSp)


    return H 
