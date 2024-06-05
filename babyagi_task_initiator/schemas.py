from pydantic import BaseModel

class InputSchema(BaseModel):
    objective: str
