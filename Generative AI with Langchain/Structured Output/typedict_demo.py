from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

person: Person = {'name': 'John Smith', 'age': "28"}
print(person)