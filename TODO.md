# To fix

- ~~Something is slow in fit analysis, investigate~~
- mean field code is too slow for big basis
- apply_H must be refactored

# Organization 

- i use Lanczos in different places, I should have a module which implement different Lanczos flavor once 
- mean field : should be a generic function, spin is only a particular case of the block case
    - add a spinblock_only possibility for mfscf then
- ~~maybe use different modules, I have too many files~~
    - lanczos 
    

# Cosmetics

- Fit is too verbose
- many stuff are too verbose

