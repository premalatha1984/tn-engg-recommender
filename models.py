from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class StudentProfile(BaseModel):
    name: Optional[str] = None
    cutoff: float = Field(..., ge=0, le=200)
    category: str  # OC/BC/MBC/SC/ST
    preferred_branches: List[str] = []
    district: str
    budget: int = Field(..., ge=0)
    first_gen: bool = False
    rural_background: bool = False
    need_hostel: bool = False

class Weights(BaseModel):
    affordability: float = 0.25
    proximity: float = 0.2
    placements: float = 0.2
    quality: float = 0.15
    rural_support: float = 0.1
    hostel: float = 0.05
    branch_priority: float = 0.05

class RecommendationRequest(BaseModel):
    profile: StudentProfile
    weights: Optional[Weights] = None
    top_k: int = 20

class Explanation(BaseModel):
    components: Dict[str, float]
    notes: List[str] = []

class RecommendationItem(BaseModel):
    rank: int
    college_code: str
    college_name: str
    district: str
    ownership: str
    program: str
    annual_fee: int
    placement_rate: float
    quality_score: float
    distance_km: float
    eligibility_margin: float
    total_score: float
    explanation: Explanation

class OptionsResponse(BaseModel):
    categories: List[str]
    branches: List[str]
    districts: List[str]
    default_weights: Weights
