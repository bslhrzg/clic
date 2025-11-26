#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <complex.h>
#include <stdio.h>
#include <unistd.h>

// Helper to wrap raw pointers into Python MemoryViews
PyObject* wrap_array(void* data, size_t size_bytes) {
    // Make memory view (Read/Write)
    return PyMemoryView_FromMemory((char*)data, size_bytes, PyBUF_WRITE);
}

int run_impmod_ed(
    char* label, char* solver_param, char* dc_param, int dc_flag,
    double complex* U_mat, double complex* hyb, double complex* h_dft,
    double complex* sig, double complex* sig_real, double complex* sig_static, double complex* sig_dc,
    double* iw, double* w,
    double complex* corr_to_spherical, double complex* corr_to_cf,
    size_t n_orb, size_t n_rot_cols, size_t n_orb_full, 
    size_t n_iw, size_t n_w,
    double eim, double tau, int verbosity,
    size_t size_real, size_t size_complex
) {

    static int call_id = 0;
    static int depth   = 0;
    call_id++;
    depth++;
    fprintf(stderr, "[C] PID=%d run_impmod_ed: call_id=%d depth=%d\n",
            (int)getpid(), call_id, depth);

    fprintf(stderr, "[C] run_impmod_ed: call_id=%d depth=%d\n", call_id, depth);

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    
    // 1. Initialize Python Interpreter (if not already running)
    if (!Py_IsInitialized()) {
        Py_Initialize();
        
        // Add current directory and the 'python' subdir to sys.path so it finds the script
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('.')"); 
        PyRun_SimpleString("sys.path.append('./python')"); 
        // Note: In production, use absolute paths or PYTHONPATH env var
    }

    // 2. Import the Module
    pName = PyUnicode_DecodeFSDefault("dummy_python_solver");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // 3. Get the function
        pFunc = PyObject_GetAttrString(pModule, "solve");

        if (pFunc && PyCallable_Check(pFunc)) {
            
            // 4. Prepare Arguments (Tuple of 22 arguments)
            pArgs = PyTuple_New(23);

            // Handle Strings (Fortran strings aren't null-terminated, be careful)
            // We assume 'label' is char[18] from Fortran.
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromStringAndSize(label, 18)); 
            //PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(solver_param)); // Assuming null term or garbage risk
            //PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(dc_param));
            PyTuple_SetItem(pArgs, 1,PyUnicode_FromStringAndSize(solver_param, 100));
            PyTuple_SetItem(pArgs, 2,PyUnicode_FromStringAndSize(dc_param, 100));
            PyTuple_SetItem(pArgs, 3, PyLong_FromLong(dc_flag));

            // Wrap Arrays as MemoryViews (Pointer + Length in bytes)
            // U_mat (n_orb^4)
            PyTuple_SetItem(pArgs, 4, wrap_array(U_mat, n_orb*n_orb*n_orb*n_orb * size_complex));
            // hyb (n_orb^2 * n_w)
            PyTuple_SetItem(pArgs, 5, wrap_array(hyb, n_orb*n_orb*n_w * size_complex));
            // h_dft (n_orb^2)
            PyTuple_SetItem(pArgs, 6, wrap_array(h_dft, n_orb*n_orb * size_complex));
            // sig (n_orb^2 * n_iw)
            PyTuple_SetItem(pArgs, 7, wrap_array(sig, n_orb*n_orb*n_iw * size_complex));
            // sig_real (n_orb^2 * n_w)
            PyTuple_SetItem(pArgs, 8, wrap_array(sig_real, n_orb*n_orb*n_w * size_complex));
            // sig_static (n_orb^2)
            PyTuple_SetItem(pArgs, 9, wrap_array(sig_static, n_orb*n_orb * size_complex));
            // sig_dc (n_orb^2)
            PyTuple_SetItem(pArgs, 10, wrap_array(sig_dc, n_orb*n_orb * size_complex));
            // iw (n_iw)
            PyTuple_SetItem(pArgs, 11, wrap_array(iw, n_iw * size_real));
            // w (n_w)
            PyTuple_SetItem(pArgs, 12, wrap_array(w, n_w * size_real));
            
            // Transform matrices
            PyTuple_SetItem(pArgs, 13, wrap_array(corr_to_spherical, n_orb*n_orb_full * size_complex));
            PyTuple_SetItem(pArgs, 14, wrap_array(corr_to_cf, n_orb*n_rot_cols * size_complex));

            // Scalars
            PyTuple_SetItem(pArgs, 15, PyLong_FromSize_t(n_orb));
            PyTuple_SetItem(pArgs, 16, PyLong_FromSize_t(n_rot_cols));
            PyTuple_SetItem(pArgs, 17, PyLong_FromSize_t(n_orb_full));
            PyTuple_SetItem(pArgs, 18, PyLong_FromSize_t(n_iw));
            PyTuple_SetItem(pArgs, 19, PyLong_FromSize_t(n_w));
            PyTuple_SetItem(pArgs, 20, PyFloat_FromDouble(eim));
            PyTuple_SetItem(pArgs, 21, PyFloat_FromDouble(tau));
            PyTuple_SetItem(pArgs, 22, PyLong_FromLong(verbosity));

            // 5. Call Python
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != NULL) {
                // Success
                Py_DECREF(pValue);
            } else {
                // Python Error
                PyErr_Print();
                return 1;
            }
        } else {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Cannot find function 'solve'\n");
            return 1;
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load 'dummy_python_solver'\n");
        return 1;
    }

    // We do NOT call Py_Finalize() because we might come back here later,
    // and MPI codes + Py_Finalize can be buggy.
    depth--;
    return 0;
}