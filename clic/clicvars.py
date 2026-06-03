
import tomllib
import numpy as np


RSPT_TO_CLICVARS = {
    "n_bath_poles": "nb",
    "Nelec_imp": "Nelec_imp",
    "num_roots": "num_roots",
    "conv_tol": "conv_tol",
    "Nmul": "Nmul",
    "temperature": "temperature",
    "NappH": "NappH",
    "lanczos_thr": "coeff_thresh",
}


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
        fit_type = "cost",
        nb_fit=1, # how many states used to fit the hybridization per correlated orbital
        warp_kind="const", # do we emphasize low energy of not
        nb=1, # how many states actually kepts per corr. orbital 
        eta_hyb=2e-3, # distance to the imag. axis used to produce the input hybridization
        eta_broad=0, # broadening distance to the imag. to help the fit
        windowing=False, # do we window the hybridization
        window_width=0.2, # with which width
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
        nelec_parity=None,

        # GREEN FUNCTIONS
        ws=None,
        iws=None,
        eta=2e-3,
        green_block_indices=None,
        L_lanczos=100,
        NappH=1,
        coeff_thresh=1e-12,
        spin_avg_sigma = False,
        green_diag_only = False,
        #IO 
        dirdump = "dump"
    ):
        
        self.M_spatial = M_spatial
        self.NF = NF
        self.M_imp = M_imp
        self.is_impurity_model = is_impurity_model
        self.imp_indices_spatial = imp_indices_spatial
        self.imp_indices_spinfull = imp_indices_spinfull
        
        self.fit_type = fit_type
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
        self.nelec_parity = nelec_parity
        self.ws = np.array([]) if ws is None else np.array(ws)
        self.iws = np.array([]) if iws is None else np.array(iws)
        self.eta = eta
        self.green_block_indices = [] if green_block_indices is None else green_block_indices 
        self.L_lanczos = L_lanczos
        self.NappH = NappH
        self.coeff_thresh = coeff_thresh
        self.spin_avg_sigma = spin_avg_sigma
        self.green_diag_only = green_diag_only

        self.dirdump = dirdump

    @classmethod
    def from_toml(cls, filename):
        try:
            with open(filename, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            print(f"WARNING: CLIC input file '{filename}' not found. Using default ClicVars.")
            return cls()

        return cls(**data)

    @classmethod
    def from_sources(cls, filename="input.toml", rspt_clic_params=None):
        clicvars = cls()
        rspt_overrides = cls._translate_rspt_params(rspt_clic_params)
        clicvars._apply_overrides(rspt_overrides, "rspt_clic_params")

        try:
            with open(filename, "rb") as f:
                toml_overrides = tomllib.load(f)
        except FileNotFoundError:
            if rspt_clic_params is None:
                print(f"WARNING: CLIC input file '{filename}' not found. Using default ClicVars.")
            else:
                print(f"WARNING: CLIC input file '{filename}' not found. Using defaults and RSPT parameters.")
            return clicvars

        overlap = sorted(set(rspt_overrides) & set(toml_overrides))
        if overlap:
            print(
                f"WARNING: CLIC input file '{filename}' overrides RSPT parameters for: "
                + ", ".join(overlap)
            )

        clicvars._apply_overrides(toml_overrides, filename)
        return clicvars

    @classmethod
    def _translate_rspt_params(cls, rspt_clic_params):
        if rspt_clic_params is None:
            return {}

        params = {}
        for rspt_key, clic_key in RSPT_TO_CLICVARS.items():
            if rspt_key in rspt_clic_params:
                params[clic_key] = rspt_clic_params[rspt_key]

        if "label" in rspt_clic_params:
            params["dirdump"] = "dump_" + rspt_clic_params["label"]

        return params

    def _apply_overrides(self, overrides, source):
        valid_keys = set(self.__dict__)
        unknown = sorted(set(overrides) - valid_keys)
        if unknown:
            raise TypeError(
                f"Unknown ClicVars option(s) in {source}: " + ", ".join(unknown)
            )

        for key, val in overrides.items():
            setattr(self, key, val)
    
    def __str__(self):
        excluded = {"ws", "iws"}
        lines = []
        for key, val in self.__dict__.items():
            if key in excluded:
                continue
            lines.append(f"{key:20s} : {val}")
        return "\n".join(lines)

    def print(self):
        print(self)


#vars = ClicVars()
#print(vars)

#vars2 = ClicVars.from_toml("input.toml")
#vars2.print()
