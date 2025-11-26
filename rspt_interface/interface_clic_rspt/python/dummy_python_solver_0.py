def solve_(label, solver_param, dc_param, dc_flag, 
          U_mat_view, hyb_view, h_dft_view, sig_view, 
          sig_real_view, sig_static_view, sig_dc_view,
          iw_view, w_view, 
          corr_to_sph_view, corr_to_cf_view,
          n_orb, n_rot, n_orb_full, n_iw, n_w, eim, tau, verbosity):

    # Strip whitespace from label for the filename
    lbl = label.strip()
    pid = os.getpid()
    
    print(f"\n [Python] --- Dumping data for Cluster: {lbl} (PID: {pid}) ---")

    # --- 1. Interpret Scalars & Strings ---
    # These are already Python objects thanks to the C-Bridge

    # --- 2. Convert Memory Views to NumPy Arrays ---
    # Crucial: Fortran arrays are Column-Major (F-order). 
    # We must specify order='F' during reshape so indices match (i, j, k) correctly.

    # 4-Index U Matrix (n_orb, n_orb, n_orb, n_orb)
    U_mat = np.frombuffer(U_mat_view, dtype=np.complex128)
    U_mat = U_mat.reshape((n_orb, n_orb, n_orb, n_orb), order='F')

    # Hybridization (n_orb, n_orb, n_w) -> Note: Interface says n_w (Real axis/Mesh)
    hyb = np.frombuffer(hyb_view, dtype=np.complex128)
    hyb = hyb.reshape((n_orb, n_orb, n_w), order='F')

    # Local Hamiltonian (n_orb, n_orb)
    h_dft = np.frombuffer(h_dft_view, dtype=np.complex128)
    h_dft = h_dft.reshape((n_orb, n_orb), order='F')

    # Self Energy Matsubara (n_orb, n_orb, n_iw)
    sig = np.frombuffer(sig_view, dtype=np.complex128)
    sig = sig.reshape((n_orb, n_orb, n_iw), order='F')

    # Self Energy Real Axis (n_orb, n_orb, n_w)
    sig_real = np.frombuffer(sig_real_view, dtype=np.complex128)
    sig_real = sig_real.reshape((n_orb, n_orb, n_w), order='F')

    # Static Self Energies (n_orb, n_orb)
    sig_static = np.frombuffer(sig_static_view, dtype=np.complex128)
    sig_static = sig_static.reshape((n_orb, n_orb), order='F')

    sig_dc = np.frombuffer(sig_dc_view, dtype=np.complex128)
    sig_dc = sig_dc.reshape((n_orb, n_orb), order='F')

    # Frequency Meshes (Real double)
    iw = np.frombuffer(iw_view, dtype=np.float64) # Size n_iw
    w = np.frombuffer(w_view, dtype=np.float64)   # Size n_w

    # Basis Transformation Matrices
    # corr_to_spherical (n_orb, n_orb_full)
    c2s = np.frombuffer(corr_to_sph_view, dtype=np.complex128)
    c2s = c2s.reshape((n_orb, n_orb_full), order='F')

    # corr_to_cf (n_orb, n_rot)
    c2cf = np.frombuffer(corr_to_cf_view, dtype=np.complex128)
    c2cf = c2cf.reshape((n_orb, n_rot), order='F')

    # --- 3. Write to HDF5 ---
    # We include PID in filename to avoid conflicts if MPI ranks write simultaneously
    filename = f"debug_solver_{lbl}_{pid}.h5"
    
    try:
        with h5py.File(filename, 'w') as f:
            # Metadata group
            g_meta = f.create_group("metadata")
            g_meta.attrs["label"] = lbl
            g_meta.attrs["solver_param"] = str(solver_param).strip()
            g_meta.attrs["dc_param"] = str(dc_param).strip()
            g_meta.attrs["dc_flag"] = dc_flag
            g_meta.attrs["n_orb"] = n_orb
            g_meta.attrs["n_rot"] = n_rot
            g_meta.attrs["n_orb_full"] = n_orb_full
            g_meta.attrs["n_iw"] = n_iw
            g_meta.attrs["n_w"] = n_w
            g_meta.attrs["eim"] = eim
            g_meta.attrs["tau"] = tau
            g_meta.attrs["verbosity"] = verbosity

            # Data group
            g_data = f.create_group("data")
            g_data.create_dataset("U_mat", data=U_mat)
            g_data.create_dataset("hyb", data=hyb)
            g_data.create_dataset("h_dft", data=h_dft)
            g_data.create_dataset("sig", data=sig)
            g_data.create_dataset("sig_real", data=sig_real)
            g_data.create_dataset("sig_static", data=sig_static)
            g_data.create_dataset("sig_dc", data=sig_dc)
            g_data.create_dataset("iw", data=iw)
            g_data.create_dataset("w", data=w)
            g_data.create_dataset("corr_to_spherical", data=c2s)
            g_data.create_dataset("corr_to_cf", data=c2cf)
        
        print(f" [Python] Successfully wrote: {filename}")
        
    except Exception as e:
        print(f" [Python] ERROR writing HDF5: {e}")

    print(" [Python] ---------------------------------------------------\n")

    return 0



import sys
import os
import numpy as np
import h5py
from mpi4py import MPI # <--- The Magic



def solve(label, solver_param, dc_param, dc_flag, 
          U_mat_view, hyb_view, h_dft_view, sig_view, 
          sig_real_view, sig_static_view, sig_dc_view,
          iw_view, w_view, 
          corr_to_sph_view, corr_to_cf_view,
          n_orb, n_rot, n_orb_full, n_iw, n_w, eim, tau, verbosity):

    # Get the Communicator from the host application
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- ONLY RANK 0 WRITES ---
    if rank != 0:
        return 0

    lbl = label.strip()
    print(f"\n [Python] --- (MPI Rank {rank}/{size}) Dumping data for Cluster: {lbl} ---")

    # --- 1. Interpret Scalars & Strings ---
    # ... (Same as before) ...

    # --- 2. Convert Memory Views to NumPy Arrays ---
    
    # 4-Index U Matrix (n_orb, n_orb, n_orb, n_orb)
    U_mat = np.frombuffer(U_mat_view, dtype=np.complex128)
    U_mat = U_mat.reshape((n_orb, n_orb, n_orb, n_orb), order='F')

    # Hybridization (n_orb, n_orb, n_w)
    hyb = np.frombuffer(hyb_view, dtype=np.complex128)
    hyb = hyb.reshape((n_orb, n_orb, n_w), order='F')

    # Local Hamiltonian (n_orb, n_orb)
    h_dft = np.frombuffer(h_dft_view, dtype=np.complex128)
    h_dft = h_dft.reshape((n_orb, n_orb), order='F')

    # Self Energy Matsubara (n_orb, n_orb, n_iw)
    sig = np.frombuffer(sig_view, dtype=np.complex128)
    sig = sig.reshape((n_orb, n_orb, n_iw), order='F')

    # Self Energy Real Axis (n_orb, n_orb, n_w)
    sig_real = np.frombuffer(sig_real_view, dtype=np.complex128)
    sig_real = sig_real.reshape((n_orb, n_orb, n_w), order='F')

    # Static Self Energies (n_orb, n_orb)
    sig_static = np.frombuffer(sig_static_view, dtype=np.complex128)
    sig_static = sig_static.reshape((n_orb, n_orb), order='F')

    sig_dc = np.frombuffer(sig_dc_view, dtype=np.complex128)
    sig_dc = sig_dc.reshape((n_orb, n_orb), order='F')

    # Frequency Meshes
    iw = np.frombuffer(iw_view, dtype=np.float64)
    w = np.frombuffer(w_view, dtype=np.float64)

    # Basis Transformation Matrices
    c2s = np.frombuffer(corr_to_sph_view, dtype=np.complex128)
    c2s = c2s.reshape((n_orb, n_orb_full), order='F')

    c2cf = np.frombuffer(corr_to_cf_view, dtype=np.complex128)
    c2cf = c2cf.reshape((n_orb, n_rot), order='F')

    # --- 3. Write to HDF5 ---
    filename = f"debug_solver_{lbl}.h5" # Removed Rank/PID since only Rank 0 writes
    
    try:
        with h5py.File(filename, 'w') as f:
            # Metadata group
            g_meta = f.create_group("metadata")
            g_meta.attrs["label"] = lbl
            g_meta.attrs["solver_param"] = str(solver_param).strip()
            g_meta.attrs["dc_param"] = str(dc_param).strip()
            g_meta.attrs["dc_flag"] = dc_flag
            g_meta.attrs["n_orb"] = n_orb
            g_meta.attrs["n_rot"] = n_rot
            g_meta.attrs["n_orb_full"] = n_orb_full
            g_meta.attrs["n_iw"] = n_iw
            g_meta.attrs["n_w"] = n_w
            g_meta.attrs["eim"] = eim
            g_meta.attrs["tau"] = tau
            g_meta.attrs["verbosity"] = verbosity

            # Data group
            g_data = f.create_group("data")
            g_data.create_dataset("U_mat", data=U_mat)
            g_data.create_dataset("hyb", data=hyb)
            g_data.create_dataset("h_dft", data=h_dft)
            g_data.create_dataset("sig", data=sig)
            g_data.create_dataset("sig_real", data=sig_real)
            g_data.create_dataset("sig_static", data=sig_static)
            g_data.create_dataset("sig_dc", data=sig_dc)
            g_data.create_dataset("iw", data=iw)
            g_data.create_dataset("w", data=w)
            g_data.create_dataset("corr_to_spherical", data=c2s)
            g_data.create_dataset("corr_to_cf", data=c2cf)
        
        print(f" [Python] Successfully wrote: {filename}")
        
    except Exception as e:
        print(f" [Python] ERROR writing HDF5: {e}")

    print(" [Python] ---------------------------------------------------\n")


    return 0