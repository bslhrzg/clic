#include <stdio.h>
#include <complex.h>
#include <stddef.h>

/* 
   We use _Complex double to match Fortran's complex(c_double_complex).
   We use size_t for integer(c_size_t).
   Arguments marked with value in Fortran are passed by value here.
   Arrays in Fortran are passed as pointers here.
*/

int run_impmod_ed(
    char* label,                 // label(18)
    char* solver_param,          // solver_param(100)
    char* dc_param,              // dc_param(100)
    int dc_flag,                 // value
    double complex* U_mat,       // array
    double complex* hyb,         // array
    double complex* h_dft,       // array
    double complex* sig,         // array
    double complex* sig_real,    // array
    double complex* sig_static,  // array
    double complex* sig_dc,      // array
    double* iw,                  // array
    double* w,                   // array
    double complex* corr_to_spherical, // array
    double complex* corr_to_cf,        // array
    size_t n_orb,                // value
    size_t n_rot_cols,           // value
    size_t n_orb_full,           // value
    size_t n_iw,                 // value
    size_t n_w,                  // value
    double eim,                  // value
    double tau,                  // value
    int verbosity,               // value
    size_t size_real,            // value
    size_t size_complex          // value
) {
    // 1. Print a message to prove we are actually inside the library
    printf("\n");
    printf(" [Dummy External Lib] ----------------------------------------\n");
    printf(" [Dummy External Lib] Function run_impmod_ed called successfully!\n");
    
    // Careful printing the label, it is not guaranteed to be null-terminated from Fortran
    printf(" [Dummy External Lib] Label: %.18s\n", label); 
    printf(" [Dummy External Lib] n_orb: %zu, n_iw: %zu\n", n_orb, n_iw);
    printf(" [Dummy External Lib] This library does absolutely nothing.\n");
    printf(" [Dummy External Lib] ----------------------------------------\n");
    printf("\n");

    // 2. Return 0 to signal "Success" to the Fortran code (er = 0)
    return 0;
}