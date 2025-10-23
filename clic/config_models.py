from pydantic import BaseModel, Field
from typing import Literal, Union

class BathConfig(BaseModel):
    nb: int = Field(..., gt=0)
    min_e: float
    max_e: float
    hybridization_V: float

class ModelParameters(BaseModel):
    type: Literal["anderson_impurity_model"]
    M_spatial: int = Field(..., gt=0)
    Nelec: int = Field(..., ge=0)
    interaction_u: float
    mu: Union[float, Literal["u/2"]]
    bath: BathConfig

class ModelConfig(BaseModel):
    model_name: str
    source_type: Literal["parametric"]
    parameters: ModelParameters

class CiMethodConfig(BaseModel):
    type: Literal["sci", "fci"]
    generator: Literal["hamiltonian_generator"]
    selector: Literal["cipsi"]
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