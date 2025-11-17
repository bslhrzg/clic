# To fix

- ~~Something is slow in fit analysis, investigate~~
- ~~mean field code is too slow for big basis~~
- ~~apply_H must be refactored~~
- ~~sci build basis from gs, instead of from num_roots, so for Ne odd, i miss a doublet~~
- revoir gfs module, trop lent:
    - ~~blocks~~ 
    - apply_hamiltonian
- revoir TGS analysis, trop lent 

# To add

- multiorb aim as new model class
- $\Delta_{approx}$ and $\Sigma$
- Green calc for Matsubara freq

# Organization 

- i use Lanczos in different places, I should have a module which implement different Lanczos flavor once 
- ~~mean field : should be a generic function, spin is only a particular case of the block case~~
    - ~~add a spinblock_only possibility for mfscf then~~
- ~~maybe use different modules, I have too many files~~
    - ~~lanczos~~
- either decide on one sci worflow, either add options
- reorganize where get_ham and everything goes


# Cosmetics

- Fit is too verbose
- many stuff are too verbose

