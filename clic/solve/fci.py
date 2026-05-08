# fci.py
import clic_clib as cc
import numpy as np
from clic.basis import basis_Np

from time import time
from clic.io_clic.io_utils import vprint 
from .diagh import get_ham, diagH

applyH=False
dodavidson=False


def do_fci(h0,U,C,M,Nelec,num_roots=1,Sz=None,verbose=True):

    basis = basis_Np.get_fci_basis(M, Nelec)

    num_roots = np.min([num_roots,len(basis)])


    if len(basis) == 0 : 
        eigvals = [0]
        psis = [cc.Wavefunction(M, basis, [1.0])]
        fci_res = {
            "Nelec": Nelec,
            "energies": eigvals,
            "wavefunctions": psis,
            "C": None,                 # or None
            "basis": basis,         # only if needed by your wf object
            }
        return fci_res


    if Sz is not None:
        #inds, blocks = partition_by_Sz(basis)    # lists of indices + Sz values
        basis, idxs0 = basis_Np.subbasis_by_Sz(basis, Sz)  # S_z = 0 sector
    print(f"fci basis size = {len(basis)}")

    t0 = time()
    H_sparse = get_ham(basis,h0,U)
    t1 = time()
    vprint(1,f"time to construct H : {t1-t0}")
    
    #eigvals, eigvecs = eigsh(H_sparse, k=num_roots, which='SA')
    eigvals,eigvecs = diagH(H_sparse, num_roots, option="arpack")
    #num_roots = np.min(num_roots,len(eigvals))
    #t2 = time()
    #vprint(1,f"time to diagonalize H : {t2-t1}")
    
    if verbose:
        Es_str = " ".join(f"{ev:.12f}" for ev in eigvals[:num_roots])
        print(f"Es=[{Es_str}]")
    
    psis = [cc.Wavefunction(M, basis, eigvecs[:,i]) for i in range(num_roots)]

    fci_res = {
            "Nelec": Nelec,
            "energies": eigvals,
            "wavefunctions": psis,
            "C": C,                 # or None
            "basis": basis,         # only if needed by your wf object
            }
    return fci_res
