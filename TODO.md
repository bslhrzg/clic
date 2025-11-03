# To fix

- Something is slow in fit analysis, investigate
- mean field code is too slow for big basis

# Organization 

- i use Lanczos in different places, I should have a module which implement different Lanczos flavor once 
- mean field : should be a generic function, spin is only a particular case of the block case
- maybe use different modules, I have too many files
    - lanczos 
    - symmetries
    - hybfit 
    - meanfield
    - solve 
    - gfs


# Cosmetics

- Fit is too verbose
- many stuff are too verbose

