from pydantic import BaseModel

class AccountNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    
