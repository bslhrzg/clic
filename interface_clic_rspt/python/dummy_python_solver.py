import sys
import os
import numpy as np
import h5py
from mpi4py import MPI # <--- The Magic


from clic import *



np.set_printoptions(precision=3, suppress=True, linewidth=300)



def prepare_from_rspt(
        n_orb,
        n_orb_full,
        hyb_rspt,h_imp_rspt,U_imp_rspt,
        n_rot_cols,
        rspt_corr_to_spherical_arr,
        rspt_corr_to_cf_arr):

    if n_rot_cols == n_orb_full and n_orb == n_orb_full:
        corr_to_spherical = rspt_corr_to_spherical_arr
        corr_to_cf = rspt_corr_to_cf_arr
    else:
        corr_to_spherical = np.empty((n_orb, 2 * n_orb_full), dtype=complex)
        corr_to_cf = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_spherical[:, :n_orb_full] = rspt_corr_to_spherical_arr
        corr_to_spherical[:, n_orb_full:] = np.roll(
            rspt_corr_to_spherical_arr, n_orb_full, axis=0
        )
        corr_to_cf[:, :n_rot_cols] = rspt_corr_to_cf_arr
        corr_to_cf[:, n_rot_cols:] = np.roll(rspt_corr_to_cf_arr, n_rot_cols, axis=0)


    print(f"h_imp shape = {h_imp_rspt.shape}")
    print(f"U_imp.shape = {U_imp_rspt.shape}")

    print("h_imp real : ")
    print(np.real(h_imp_rspt))

    print(f"n_orb = {n_orb}")
    print(f"corr_to_cf shape = {corr_to_cf.shape}")


    h_imp = np.ascontiguousarray(h_imp_rspt)
    hyb = np.ascontiguousarray(np.moveaxis(hyb_rspt, -1, 0))

    U_imp = basis_change_U(U_imp_rspt,corr_to_cf)

    return hyb,h_imp,U_imp
 
CALL_COUNT=0

def solve(label, solver_param_, dc_param, dc_flag, 
          U_mat_view, hyb_view, h_dft_view, sig_view, 
          sig_real_view, sig_static_view, sig_dc_view,
          iw_view, w_view, 
          corr_to_sph_view, corr_to_cf_view,
          n_orb, n_rot, n_orb_full, n_iw, n_w, eim, tau, verbosity):

    global CALL_COUNT
    CALL_COUNT += 1
    print(f"\n[Python] ENTER solve(), CALL_COUNT={CALL_COUNT}")

    # Get the Communicator from the host application
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ------------------------------------------------------------------
    # 0. Create NumPy wrappers for the output buffers on *all* ranks
    # ------------------------------------------------------------------
    # These arrays are views into the RSPT-provided memory.
    sig = np.frombuffer(sig_view, dtype=np.complex128)
    sig = sig.reshape((n_orb, n_orb, n_iw), order='F')

    sig_real = np.frombuffer(sig_real_view, dtype=np.complex128)
    sig_real = sig_real.reshape((n_orb, n_orb, n_w), order='F')

    sig_static = np.frombuffer(sig_static_view, dtype=np.complex128)
    sig_static = sig_static.reshape((n_orb, n_orb), order='F')

    sig_dc = np.frombuffer(sig_dc_view, dtype=np.complex128)
    sig_dc = sig_dc.reshape((n_orb, n_orb), order='F')  # if you ever need it

    iw = np.frombuffer(iw_view, dtype=np.float64)
    w  = np.frombuffer(w_view,  dtype=np.float64)

    # Placeholders for results (will be filled on rank 0)
    res_static = None
    res_sigma = None
    res_sigma_iw = None

    # --- ONLY RANK 0 WRITES ---
    if rank == 0:

        lbl = label.strip()
        print(f"\n [Python] --- (MPI Rank {rank}/{size}) Dumping data for Cluster: {lbl} ---")

        # --- 1. Interpret Scalars & Strings ---
        # ... (Same as before) ...

        # --- 2. Convert Memory Views to NumPy Arrays ---

        print(f"Solver params : {solver_param_.strip()}")

        solver_param = solver_param_.strip().split()
        print(f"solver_param = {solver_param}")
        if len(solver_param) < 2 : 
            raise ValueError(f"Expected at least 2 parameters, got {solver_param}")

        clic_params = {}
        clic_params["n_bath_poles"] = int(solver_param[0]) 
        clic_params["Nelec_imp"]    = int(solver_param[1])


        clic_params["num_roots"] = int(solver_param[2]) if len(solver_param) > 2 else 4
        clic_params["temperature"]  = float(solver_param[3]) if len(solver_param) > 3 else 5
        clic_params["NappH"] = int(solver_param[4]) if len(solver_param) > 4 else 1

        clic_params["conv_tol"] = float(solver_param[5]) if len(solver_param) > 5 else 5e-4
        clic_params["Nmul"] = float(solver_param[6]) if len(solver_param) > 6 else None 
        clic_params["lanczos_thr"] = float(solver_param[7]) if len(solver_param) > 7 else 1e-5 




# If you have NappH or more parameters, continue...
        #n_bath_poles, Nelec_imp, num_roots, temperature, NappH = solver_param.strip()
        #print(f"n_bath_poles = {n_bath_poles}, Nelec_imp = {Nelec_imp}, \
        #      num_roots = {num_roots}, temperature = {temperature}, NappH = {NappH}")
        
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


        hyb_clic,h_imp_clic,U_imp_clic = prepare_from_rspt(n_orb,
            n_orb_full,
            hyb,
            h_dft,
            U_mat,
            n_rot,
            c2s,
            c2cf)
        
        res_static,res_sigma,res_sigma_iw = dmft_step(
            w,iw,hyb_clic,h_imp_clic,U_imp_clic,clic_params)
    

    # ------------------------------------------------------------------
    # 2. Broadcast results to all ranks
    # ------------------------------------------------------------------
    # On non-root ranks, res_* is currently None; that's fine, bcast will fill it.
    res_static = comm.bcast(res_static, root=0)
    res_sigma = comm.bcast(res_sigma, root=0)
    res_sigma_iw = comm.bcast(res_sigma_iw, root=0)

    # ------------------------------------------------------------------
    # 3. Write results back into RSPT buffers on *all* ranks
    # ------------------------------------------------------------------
    # 1. Static Self Energy: (n_orb, n_orb)
    sig_static[:] = res_static[:]

    # 2. Real Axis Self Energy
    if res_sigma.shape[0] == n_w:
        # (w, orb, orb) -> (orb, orb, w)
        print("Coucou, on modifie sig_real")
        sig_real[:] = np.moveaxis(res_sigma, 0, -1)
    else:
        # Already (orb, orb, w)
        sig_real[:] = res_sigma[:]

    # 3. Matsubara Self Energy
    if res_sigma_iw.shape[0] == n_iw:
        # (iw, orb, orb) -> (orb, orb, iw)
        print("Coucou, on modifie sig")
        sig[:] = np.moveaxis(res_sigma_iw, 0, -1)
    else:
        sig[:] = res_sigma_iw[:]

    er = 0
    print(f"coucou, returning {er} here")
    print("*" * 108)

    return er