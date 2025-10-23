# config_models.py
from pydantic import BaseModel, Field
from typing import Literal, Union, List

class BathConfig(BaseModel):
    nb: int = Field(..., gt=0)
    min_e: float
    max_e: float
    hybridization_V: float

class AimParameters(BaseModel):
    type: Literal["anderson_impurity_model"]
    M_spatial: int = Field(..., gt=0)
    Nelec: int = Field(..., ge=0)
    interaction_u: float
    mu: Union[float, Literal["u/2"]]
    bath: BathConfig

class FileDataSource(BaseModel):
    type: Literal["from_file"]
    filepath: str
    Nelec: int = Field(..., ge=0)
    spin_structure: Literal["alpha_first", "interleaved"] = "alpha_first"


class ModelConfig(BaseModel):
    model_name: str
    parameters: Union[AimParameters, FileDataSource] = Field(..., discriminator='type')

class CiMethodConfig(BaseModel):
    type: Literal["sci", "fci"]
    generator: Literal["hamiltonian_generator"]
    selector: Literal["cipsi"]
    num_roots: int = Field(1, gt=0) 
    max_iter: int = Field(..., gt=0)
    conv_tol: float = Field(..., gt=0)
    prune_thr: float = Field(..., ge=0)
    Nmul: Union[float, None] = None

class SolverParameters(BaseModel):
    basis_prep_method: Literal["none", "rhf", "rhf_no", "dbl_chain"]
    ci_method: CiMethodConfig

class OutputConfig(BaseModel):
    ground_state_file: str

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