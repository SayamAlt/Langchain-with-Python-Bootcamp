from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Sayam Kumar'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=7.5,description='A decimal value representing the CGPA of the student')

new_student = {'age': '24', 'email': 'abc@gamil.com', 'cgpa': 6.2}

student = Student(**new_student)

student_dict = dict(student)

student_json = student.model_dump_json()
print(student_json)