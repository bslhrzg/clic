# config_models.py
from typing import Literal, Union, List, Optional 
from pydantic import BaseModel, Field, root_validator, ConfigDict


class BathConfig(BaseModel):
    nb: int = Field(..., gt=0)
    min_e: float
    max_e: float
    hybridization_V: float

class AimParameters(BaseModel):
    type: Literal['anderson_impurity_model']
    M_spatial: int
    M_imp: int
    Nelec_imp: int
    Nelec: Optional[int] = None
    interaction_u: float
    mu: Union[float, Literal["u/2"]]
    bath: BathConfig

    @root_validator(pre=False, skip_on_failure=True)
    def check_sizes(cls, values):
        m_imp, m_spatial = values.get('M_imp'), values.get('M_spatial')
        if m_imp is not None and m_spatial is not None and m_imp > m_spatial:
            raise ValueError("M_imp (impurity orbitals) cannot be larger than M_spatial (total orbitals).")
        return values

class FileDataSource(BaseModel):
    type: Literal["from_file"]
    filepath: str
    Nelec: int = Field(..., ge=0)
    spin_structure: Literal["alpha_first", "interleaved"] = "interleaved"

class FileImpurityModelParameters(BaseModel):
    """Defines an impurity model where integrals are loaded from a file."""
    type: Literal["impurity_from_file"]
    filepath: str
    M_imp: int                 # Number of impurity spatial orbitals
    Nelec_imp: int             # Target number of electrons ON THE IMPURITY
    Nelec: Optional[int] = None # Optional: TOTAL number of electrons
    spin_structure: Literal["alpha_first", "interleaved"] = "interleaved"


class ModelConfig(BaseModel):
    model_name: str
    parameters: Union[
        AimParameters, 
        FileDataSource, 
        FileImpurityModelParameters] = Field(..., discriminator='type')

class CiMethodConfig(BaseModel):
    type: Literal["sci", "fci"] = "sci"
    generator: Literal["hamiltonian_generator"]
    selector: Literal["cipsi"]
    num_roots: int = Field(1, gt=0) 
    max_iter: int = Field(2, gt=-1)
    conv_tol: float = Field(1e-6, gt=0)
    prune_thr: float = Field(1e-7, ge=0)
    Nmul: Union[float, None] = None

class SolverParameters(BaseModel):
    basis_prep_method: Literal["none", "rhf", "rhf_no", "bath_no","dbl_chain"]
    use_no: Literal["none","no0","no"] = "none"
    ci_method: CiMethodConfig
    nelec_range: Union[tuple[int, int], Literal["auto"], None] = None
    initial_temperature: float = 10.0 

class OutputConfig(BaseModel):
    ground_state_file: str = "clic_solve_results.h5"

class SolverConfig(BaseModel):
    model_file: str
    solver: SolverParameters
    output: OutputConfig

# --- Green's Function Configuration ---

class GreenFunctionParameters(BaseModel):
    omega_mesh: List[Union[int, float]]
    eta: float = Field(..., gt=0)
    block_indices: Union[Literal["impurity"], List[int]]

class LanczosParameters(BaseModel):
    L: int = Field(..., gt=0)
    NappH: int = Field(..., ge=0)
    coeff_thresh: float = Field(..., ge=0)

class GfOutputConfig(BaseModel):
    gf_data_file: str
    plot_file: str | None = None

class GfConfig(BaseModel):
    ground_state_file: str
    green_function: GreenFunctionParameters
    lanczos: LanczosParameters
    output: GfOutputConfig