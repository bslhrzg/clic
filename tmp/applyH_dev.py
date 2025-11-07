import numpy as np
import h5py
from itertools import combinations
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

import clic_clib as qc

from typing import Dict, Iterable, Tuple, List, Set, Union, Callable
import numpy as np

SUType = Dict[int, Union[Iterable[int], Dict[int, Set[int]]]]
DType  = Dict[Tuple[int, int], Iterable[Tuple[int, int]]]


def build_Sh0(h0: np.ndarray, threshold: float) -> Dict[int, List[int]]:
    """
    Sh0[i] = list of j such that j != i and |h0[j,i]| > threshold.
    Uses column-based selection since singles from i->j use h0[j,i].
    """
    K = h0.shape[0]
    assert h0.shape == (K, K)
    Sh0: Dict[int, List[int]] = {}
    abs_h0 = np.abs(h0)
    for i in range(K):
        # all j with |h0[j,i]| > thr and j != i
        js = np.flatnonzero(abs_h0[:, i] > threshold).tolist()
        if i in js:
            js.remove(i)
        if js:
            Sh0[i] = js
    return Sh0

def build_SU(U: np.ndarray, threshold: float, detailed: bool = True):
    """
    SU in detailed mode:
      SU[i][j] = set of spectator p such that at least one of
        U[j,p,p,i], U[j,p,i,p], U[p,j,p,i], U[p,j,i,p]
      exceeds threshold in absolute value.

    If detailed=False, returns SU[i] = list of j with any spectator p satisfying the condition.
    """
    K = U.shape[0]
    assert U.shape == (K, K, K, K)
    SU: Dict[int, Dict[int, Set[int]]] = {} if detailed else {}
    absU = np.abs(U)

    # We vectorize over p for each fixed (i,j)
    for i in range(K):
        inner = {} if detailed else None
        for j in range(K):
            if j == i:
                continue
            # gather four p-dependent 1D arrays
            # shapes: (K,)
            a = absU[j, :, :, i].diagonal()          # U[j,p,p,i]
            b = absU[j, :, i, :].diagonal()          # U[j,p,i,p]
            c = absU[:, j, :, i].diagonal()          # U[p,j,p,i]
            d = absU[:, j, i, :].diagonal()          # U[p,j,i,p]

            # p such that any term > threshold
            ok_p = np.flatnonzero((a > threshold) | (b > threshold) | (c > threshold) | (d > threshold))
            if ok_p.size == 0:
                continue

            if detailed:
                inner[j] = set(ok_p.tolist())
            else:
                # simple mode collects only j, ignoring which p triggered it
                inner = inner or []
                inner.append(j)

        if detailed and inner:
            SU[i] = inner
        elif not detailed and inner:
            # simple mode: SU[i] = sorted unique js
            SU[i] = sorted(set(inner))
    return SU

def build_D(U: np.ndarray, threshold: float) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    D[(i,j)] with i<j maps to a list of (k,l) such that
      k != l, {k,l} ∩ {i,j} = ∅, and any of
        U[k,l,i,j], U[l,k,i,j], U[k,l,j,i], U[l,k,j,i]
      exceeds threshold in absolute value.

    Notes
    -----
    1) We do not assume any antisymmetry or 8-fold symmetry of U.
       If your U has standard chemists' or physicists' symmetries,
       this screen is conservative and safe.
    2) If you want to shrink D further, you can also include
       the Hermitian-conjugate checks on (i,j,k,l), but KL will
       take care of the exact sign and value anyway.
    """
    K = U.shape[0]
    assert U.shape == (K, K, K, K)
    absU = np.abs(U)

    D: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    # Precompute pair mask for each (i,j) indicating which (k,l) couple
    for i in range(K):
        for j in range(i + 1, K):
            # base 2D slices over (k,l)
            A = absU[:, :, i, j]      # U[k,l,i,j]
            B = absU[:, :, j, i]      # U[k,l,j,i]
            # also need the swapped first pair (l,k,*): we can test by transposing A and B
            mask = (A > threshold) | (B > threshold) | (A.T > threshold) | (B.T > threshold)

            if not np.any(mask):
                continue

            # remove illegal indices: k==l, or any overlap with i or j
            # Start from candidate list
            ks, ls = np.nonzero(mask)
            good = []
            for k, l in zip(ks, ls):
                if k == l:
                    continue
                if k == i or k == j or l == i or l == j:
                    continue
                good.append((k, l))

            if good:
                D[(i, j)] = good
    return D


# Helpers for the occupied-list representation
def _as_key(occ_list: List[int]) -> Tuple[int, ...]:
    # determinants are keyed by the sorted tuple of occupied orbitals
    return tuple(sorted(occ_list))

def _occ_sets_from_list(det: List[int], K: int) -> Tuple[Set[int], Set[int]]:
    occ = set(det)
    emp = set(range(K)) - occ
    return occ, emp

def _single_exc(det: List[int], i: int, j: int) -> List[int]:
    # remove occupied i, add empty j, keep sorted
    # assumes i in det and j not in det
    new_occ = [x for x in det if x != i]
    new_occ.append(j)
    new_occ.sort()
    return new_occ

def _double_exc(det: List[int], i: int, j: int, k: int, l: int) -> List[int]:
    # remove occupied i,j, add empty k,l, keep sorted
    # assumes i,j in det and k,l not in det and all distinct
    rem = set(det)
    rem.remove(i)
    rem.remove(j)
    rem_list = sorted(rem)
    rem_list.extend([k, l])
    rem_list.sort()
    return rem_list

def _norm_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)

# Table driven iterators with screening against the current occupations
def _iter_singles_from_Sh0(Sh0: Dict[int, Iterable[int]],
                           occ: Set[int], emp: Set[int]) -> Iterable[Tuple[int, int]]:
    for i in occ:
        for j in Sh0.get(i, ()):
            if j in emp and j != i:
                yield (i, j)

def _iter_singles_from_SU(SU: SUType,
                          occ: Set[int], emp: Set[int]) -> Iterable[Tuple[int, int]]:
    for i in occ:
        targets = SU.get(i)
        if not targets:
            continue
        if isinstance(targets, dict):
            # detailed: i -> {j: set(p)}
            for j, pset in targets.items():
                if j in emp and j != i:
                    # require at least one spectator p currently occupied
                    if any(p in occ for p in pset):
                        yield (i, j)
        else:
            # simple: i -> iterable of j
            for j in targets:
                if j in emp and j != i:
                    yield (i, j)

def _iter_doubles_from_D(D: DType,
                         occ: Set[int], emp: Set[int]) -> Iterable[Tuple[int, int, int, int]]:
    occ_sorted = sorted(occ)
    for a in range(len(occ_sorted)):
        i = occ_sorted[a]
        for b in range(a + 1, len(occ_sorted)):
            j = occ_sorted[b]
            key = _norm_pair(i, j)
            for k, l in D.get(key, ()):
                if k == l:
                    continue
                # both targets must be empty and disjoint from {i,j}
                if (k in emp) and (l in emp) and (k not in (i, j)) and (l not in (i, j)):
                    yield (i, j, k, l)

def apply_H_on_det(Sh0: Dict[int, Iterable[int]],
                   SU: SUType,
                   D: DType,
                   det: List[int],
                   *,
                   KL: Callable[[List[int], List[int], int, np.ndarray, np.ndarray], float],
                   K: int,
                   h0: np.ndarray,
                   U: np.ndarray) -> Tuple[List[List[int]], List[float]]:
    """
    Apply H to |det> where det is a list of occupied orbitals.

    Returns:
      dets, amps
    including the diagonal element first.
    """
    occ, emp = _occ_sets_from_list(det, K)

    out_dets: List[List[int]] = []
    out_amps: List[float] = []
    seen: Set[Tuple[int, ...]] = set()

    # 1) diagonal
    diag_amp = KL(det, det, K, h0, U)
    out_dets.append(det)
    out_amps.append(diag_amp)
    seen.add(_as_key(det))

    # 2) singles from h0
    for i, j in _iter_singles_from_Sh0(Sh0, occ, emp):
        newd = _single_exc(det, i, j)
        key = _as_key(newd)
        if key in seen:
            continue
        seen.add(key)
        amp = KL(det, newd, K, h0, U)
        if amp != 0.0:
            out_dets.append(newd)
            out_amps.append(amp)

    # 3) singles from U
    for i, j in _iter_singles_from_SU(SU, occ, emp):
        newd = _single_exc(det, i, j)
        key = _as_key(newd)
        if key in seen:
            continue
        seen.add(key)
        amp = KL(det, newd, K, h0, U)
        if amp != 0.0:
            out_dets.append(newd)
            out_amps.append(amp)

    # 4) doubles from U
    for i, j, k, l in _iter_doubles_from_D(D, occ, emp):
        newd = _double_exc(det, i, j, k, l)
        key = _as_key(newd)
        if key in seen:
            continue
        seen.add(key)
        amp = KL(det, newd, K, h0, U)
        if amp != 0.0:
            out_dets.append(newd)
            out_amps.append(amp)

    return out_dets, out_amps



def load_integrals_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        h0 = f['h0'][:]
        U = f['U'][:]
        Ne = int(f.attrs['Ne'])
        Enuc = float(f.attrs['Enuc'])
        K = h0.shape[0]
        M = K // 2
    return h0, U, Ne, Enuc, M, K


def calculate_hf_energy_python(h0, U, Ne, Enuc):
    """Calculates HF energy from raw interleaved integrals, proven to be correct."""
    occupied_indices = range(Ne)
    e1 = sum(h0[i, i] for i in occupied_indices)
    e2 = sum(
        (U[i, j, i, j] - U[i, j, j, i])
        for i in occupied_indices
        for j in occupied_indices
    )
    return e1 + 0.5 * e2 + Enuc

def get_fci_basis(num_spatial, num_electrons):
    """Generates a sorted list of SlaterDeterminant objects for the FCI space."""
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        basis_dets.append(qc.SlaterDeterminant(num_spatial, occ_a, occ_b))
    return sorted(basis_dets)

def transform_integrals_interleaved_to_alphafirst(h0_int, U_int, M):
    """
    Transforms integrals from spin-interleaved to AlphaFirst ordering.
    The underlying notation (i.e., meaning of V[p,q,r,s]) is preserved.
    """
    K = 2 * M
    
    # af_map[i_af] gives the corresponding index in the original interleaved basis
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

def test_h2o_fci_energy():
    print("--- Testing FCI Ground State Energy of H2O/STO-3G ---")
    
    # 1. Load data
    h0, U, Ne, Enuc, M, K = load_integrals_from_h5("h2o_h0U.h5")
    
    # 2. Transform integrals to AlphaFirst basis
    print("\nTransforming integrals to AlphaFirst basis...")
    h0, U = transform_integrals_interleaved_to_alphafirst(
        h0, U, M
    )
    
    # 3. CRITICAL: Ensure arrays are C-contiguous and have the correct dtype
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    U =  np.ascontiguousarray(U, dtype=np.complex128)


    # 4. Generate the FCI basis
    print("\nGenerating FCI basis...")
    fci_basis = get_fci_basis(M, Ne)
    print(f"  FCI basis size for ({M}o, {Ne}e) = {len(fci_basis)} determinants.")
    
    # 5. Build the full FCI Hamiltonian matrix
    print("\nBuilding FCI Hamiltonian matrix with C++ kernel...")
    H_sparse = qc.build_hamiltonian_openmp(fci_basis, h0, U)
    print(f"  Hamiltonian construction complete. Matrix shape: {H_sparse.shape}")


    


    #============================================================================
    #============================================================================
    # The Test: Apply H to the Hartree-Fock state and compare results
    #============================================================================

    Hfci = H_sparse.toarray()

    # The HF state is | up, up, down, down > = | a0, a1, b0, b1 >
    hf_det = qc.SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))
    wfhf = qc.Wavefunction(M, [hf_det], [1.0])

    def det_to_occ(det,M):
        occ = det.alpha_occupied_indices() + [i + M for i in det.beta_occupied_indices()]
        return occ
    
    def occ_to_det(occ,M):
        occa = [o for o in occ if o < M]
        occb = [o-M for o in occ if o >= M]
        det = qc.SlaterDeterminant(M, occa, occb)
        return det 
    
    occhf = det_to_occ(hf_det,M)
    #
    K = h0.shape[0]
    thr = 1e-10
    Sh0 = build_Sh0(h0, thr)
    SU  = build_SU(U, thr, detailed=True)
    D   = build_D(U, thr)

    # then use the apply_H_on_det we wrote earlier
    occs, amps = apply_H_on_det(Sh0, SU, D, occhf, KL=qc.KL, K=K, h0=h0, U=U)

    Hhf_basis = [occ_to_det(o,M) for o in occs]
    # Apply your new function
    # Using tol=0 ensures we don't miss any connections
    #Hhf_new = qc.apply_hamiltonian(wfhf, h0, U, 0, 0)
    Hhf_new = qc.Wavefunction(M,Hhf_basis,amps)

    #print("\nBuilding screened Hamiltonian tables with C++...")
    thr = 1e-10  # Use a small threshold for screening
    screened_H = qc.build_screened_hamiltonian(h0, U, thr)

    # Apply the Hamiltonian using the new C++ function
    print("Applying Hamiltonian to HF state with new C++ kernel...")
    Hhf_new = qc.apply_hamiltonian(wfhf, screened_H, h0, U, 0) # tol_element=0

    # Find the index of the HF determinant in the FCI basis
    try:
        hf_index = fci_basis.index(hf_det)
    except ValueError:
        print("Error: HF determinant not found in FCI basis.")
        exit()

    print(f"len gen = {len(Hhf_new.get_basis())}")
    print(f"<hf|H|hf> = {Hhf_new.amplitude(hf_det)}")

    # The reference vector is the column of H corresponding to the HF state
    hf_column_ref = Hfci[:, hf_index]

    print("\n--- Comparing H|HF> results ---")
    print(f"{'Determinant':<45s} {'Reference':>15s} {'Calculated':>15s} {'abs. diff':>15s} {'Status'}")
    print("-" * 85)

    n_mismatch = 0
    tol = 1e-9

    # Iterate through the entire FCI space to check all possible connections
    for i, det_i in enumerate(fci_basis):
        cref = hf_column_ref[i]
        if np.abs(cref) > 1e-12:
            # Get the calculated amplitude using your bound function
            amp_new = Hhf_new.amplitude(det_i)

            # Check for discrepancies
            is_zero_ref = abs(cref) < tol
            is_zero_new = abs(amp_new) < tol
            
            status = ""
            # Only print entries that should be non-zero or where there's a mismatch
            if not is_zero_ref or not is_zero_new:
                if is_zero_ref and not is_zero_new:
                    status = "!! SPURIOUS !!" # New function created a connection that shouldn't exist
                    n_mismatch += 1
                elif not is_zero_ref and is_zero_new:
                    status = "!! MISSING !!"  # New function missed a connection
                    n_mismatch += 1
                elif abs(cref - amp_new) > tol:
                    status = "!! WRONG VALUE !!" # Value/sign mismatch
                    n_mismatch += 1
                else:
                    status = "OK"

                print(f"{str(det_i):<45s} {cref:15.8f} {amp_new:15.8f} {np.abs(cref-amp_new)} {status}")

    print("-" * 85)
    if n_mismatch == 0:
        print("SUCCESS: All calculated matrix elements match the reference.")
    else:
        print(f"FAILURE: Found {n_mismatch} mismatched matrix elements.")

    # Additionally, let's check for any determinants generated that are NOT in the FCI space (should be none)
    fci_basis_set = set(fci_basis)
    spurious_dets_outside_fci = []
    for det_new in Hhf_new.get_basis():
        if det_new not in fci_basis_set:
            spurious_dets_outside_fci.append(det_new)

    if spurious_dets_outside_fci:
        print("\n!! CRITICAL ERROR: Generated determinants outside the FCI space:")
        for d in spurious_dets_outside_fci:
            print(f"  - {d}")
    

    #============================================================================



    # 6. Diagonalize to find the lowest eigenvalue
    print("\nDiagonalizing FCI Hamiltonian...")
    electronic_eigenvalues, _ = eigsh(H_sparse, k=4, which='SA')
    #electronic_eigenvalues, _ = eigh(H_sparse.toarray())
    electronic_gs_energy = electronic_eigenvalues[0]
    
    # 7. Calculate total energy and validate
    total_gs_energy = electronic_gs_energy + Enuc
    print(f"\n  Lowest electronic energy = {electronic_gs_energy:.8f} Hartree")
    print(f"  Total ground state energy (Electronic + Nuclear) = {total_gs_energy:.8f} Hartree")
    
    # Reference value for FCI/STO-3G water with these integrals
    reference_fci_energy = -75.0232909847239
    print(f"  Reference FCI energy = {reference_fci_energy:.8f} Hartree")
    
    np.testing.assert_allclose(total_gs_energy, reference_fci_energy, atol=1e-6)
    print("\n✅✅✅ FINAL SUCCESS: Calculated FCI energy matches the reference value.")

if __name__ == "__main__":
    # You can keep the old test function if you want, but this is the main one
    test_h2o_fci_energy()

