from pydantic import BaseModel, Field


class InputPoem(BaseModel):
    author: str
    created: str
    header: str
    formula: str
    meter: str
    clausula: str
    feet: str
    path: str


class SyllableMasks(BaseModel):
    linguistic_accent_mask: list[bool]  # as marked by accentuator
    poetic_accent_mask: list[bool]  # as marked in corpus
    last_in_word_mask: list[bool]


class Meter(BaseModel):
    meter: str
    feet: int
    clausula: str
    unstable: bool = Field(default=False)


class Line(BaseModel):
    # строка может содержать несколько метров: например, в случае цезурного разделения строки
    meters: list[Meter]
    # слог, после которого располагается цезура. -1, если цезура отсутствует
    caesura: int
    syllable_masks: SyllableMasks


class OutputPoem(BaseModel):
    path: str
    lines: list[Line]
