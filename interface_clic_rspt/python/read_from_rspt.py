import h5py
import numpy as np
from types import SimpleNamespace

def load_solver_data(filename):
    """
    Reads the HDF5 debug archive produced by the dummy solver.
    
    Returns:
        SimpleNamespace: An object containing all arguments as attributes.
                         (e.g., data.U_mat, data.label, data.n_orb)
    """
    data = {}
    
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Loading data from: {filename}")
            
            # --- 1. Read Metadata (Scalars & Strings) ---
            meta = f['metadata'].attrs
            
            # Strings might need decoding if h5py stored them as bytes
            def decode_if_bytes(x):
                return x.decode('utf-8') if isinstance(x, bytes) else x

            data['label'] = decode_if_bytes(meta['label'])
            data['solver_param'] = decode_if_bytes(meta['solver_param'])
            data['dc_param'] = decode_if_bytes(meta['dc_param'])
            
            # Scalars
            data['dc_flag'] = int(meta['dc_flag'])
            data['n_orb'] = int(meta['n_orb'])
            data['n_rot'] = int(meta['n_rot'])
            data['n_orb_full'] = int(meta['n_orb_full'])
            data['n_iw'] = int(meta['n_iw'])
            data['n_w'] = int(meta['n_w'])
            data['eim'] = float(meta['eim'])
            data['tau'] = float(meta['tau'])
            data['verbosity'] = int(meta['verbosity'])

            # --- 2. Read DataArrays ---
            # using [()] reads the dataset into a numpy array in memory immediately
            g_data = f['data']
            
            data['U_mat'] = g_data['U_mat'][()]
            data['hyb'] = g_data['hyb'][()]
            data['h_dft'] = g_data['h_dft'][()]
            data['sig'] = g_data['sig'][()]
            data_sig = g_data['sig'][()] # Keep raw for inspection
            data['sig_real'] = g_data['sig_real'][()]
            data['sig_static'] = g_data['sig_static'][()]
            data['sig_dc'] = g_data['sig_dc'][()]
            
            data['iw'] = g_data['iw'][()]
            data['w'] = g_data['w'][()]
            
            data['corr_to_spherical'] = g_data['corr_to_spherical'][()]
            data['corr_to_cf'] = g_data['corr_to_cf'][()]

            print("Data loaded successfully.")

    except OSError:
        print(f"Error: Could not open file {filename}")
        return None
    except KeyError as e:
        print(f"Error: Missing key in HDF5 file: {e}")
        return None

    # Convert dictionary to a SimpleNamespace for dot-access (data.U_mat)
    return SimpleNamespace(**data)

# --- Example Usage ---
if __name__ == "__main__":
    #filename = "debug_solver_Ce4f.h5"
    #D = load_solver_data(filename)
    
    #if D:
    #     print(f"Label: {D.label}")
    #     print(f"U Matrix shape: {D.U_mat.shape}")
    #     print(f"Hybridization shape: {D.hyb.shape}")
    #     print(f"First Matsubara Freq: {D.iw[0]}")
    pass