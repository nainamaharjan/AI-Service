"""
   Created by: Naina Maharjan
   Created on: 2024-01-15
"""
from pydantic import BaseModel
from enum import Enum





class TaskType(str,Enum):
    GRAMMAR_CHECK = "GRAMMAR_CHECK"
    PROFESSIONAL = "PROFESSIONAL"
    CASUAL = "CASUAL"
    SHORTEN = "SHORTEN"
    ELABORATE = "ELABORATE"

class Prompt(BaseModel):
    system_prompt: str = None
    system_instruction: str = None
    response: str = None
    task_type:TaskType = None
