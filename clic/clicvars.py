
import tomllib
import numpy as np


class ClicVars:
    """
    Holds all relevant parameters.
    """

    def __init__(
        self,

        # DEGREES OF FREEDOM
        M_spatial=2, # Number of spatial orbitals
        NF=4, # 2*M_spatial
        M_imp=1, # number of spatial impurity orbitals
        is_impurity_model=True,
        imp_indices_spatial = None,
        imp_indices_spinfull = None,

        # HYBRIDIZATION FIT
        nb_fit=1, # how many states used to fit the hybridization per correlated orbital
        warp_kind="none", # do we emphasize low energy of not
        nb=1, # how many states actually kepts per corr. orbital 
        eta_hyb=1e-2, # distance to the imag. axis used to produce the input hybridization
        eta_broad=1e-2, # broadening distance to the imag. to help the fit
        windowing=False, # do we window the hybridization
        window_width=0.0, # with which width
        window_pos=0.0, # where
        diag_fit=True, # do we fit only the diagonal components of the hyb
        freeze_bath=False, # if True, look for an existing fit and load that

        # SOLVER SETTINGS
        basis_prep_method="none", # do we rotate the initial basis 
        ci_type="sci", # which kind of ci  
        num_roots=14, # how many states computed in each electron sector
        max_iter=3, # 
        conv_tol=1e-8, 
        prune_thr=1e-12,
        Nmul=None, # 

        # FOCK SPACE
        Nelec_imp=1, # how many electron do we expect in the impurity
        temperature=5.0, #  
        Nelec_target=None, # 
        nelec_range="auto",

        # GREEN FUNCTIONS
        ws=None,
        iws=None,
        eta=1e-2,
        green_block_indices=None,
        L_lanczos=100,
        NappH=1,
        coeff_thresh=1e-12,
    ):
        
        self.M_spatial = M_spatial
        self.NF = NF
        self.M_imp = M_imp
        self.is_impurity_model = is_impurity_model
        self.imp_indices_spatial = imp_indices_spatial
        self.imp_indices_spinfull = imp_indices_spinfull

        self.nb_fit = nb_fit
        self.warp_kind = warp_kind
        self.nb = nb
        self.eta_hyb = eta_hyb
        self.eta_broad = eta_broad
        self.windowing = windowing
        self.window_width = window_width
        self.window_pos = window_pos
        self.diag_fit = diag_fit
        self.freeze_bath = freeze_bath

        self.basis_prep_method = basis_prep_method
        self.ci_type = ci_type
        self.num_roots = num_roots
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.prune_thr = prune_thr
        self.Nmul = Nmul

        self.Nelec_imp = Nelec_imp
        self.temperature = temperature
        self.Nelec_target = Nelec_target
        self.nelec_range = [] if nelec_range is None else nelec_range

        self.ws = np.array([]) if ws is None else np.array(ws)
        self.iws = np.array([]) if iws is None else np.array(iws)
        self.eta = eta
        self.green_block_indices = [] if green_block_indices is None else green_block_indices 
        self.L_lanczos = L_lanczos
        self.NappH = NappH
        self.coeff_thresh = coeff_thresh

    @classmethod
    def from_toml(cls, filename):
        with open(filename, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def __str__(self):
        lines = []
        for key, val in self.__dict__.items():
            lines.append(f"{key:20s} : {val}")
        return "\n".join(lines)

    def print(self):
        print(self)


#vars = ClicVars()
#print(vars)

#vars2 = ClicVars.from_toml("input.toml")
#vars2.print()