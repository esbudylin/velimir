from enum import IntEnum

import bitarray.util as bu

from bitarray import bitarray
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

    def encode(self):
        def serialize(mask):
            return bu.serialize(bitarray(mask))

        return [
            serialize(self.linguistic_accent_mask),
            serialize(self.poetic_accent_mask),
            serialize(self.last_in_word_mask),
        ]

    @classmethod
    def decode(cls, data):
        def deserialize(b):
            ba = bu.deserialize(b)
            return ba.tolist()

        return cls(
            linguistic_accent_mask=deserialize(data[0]),
            poetic_accent_mask=deserialize(data[1]),
            last_in_word_mask=deserialize(data[2]),
        )


class CodeIntEnum(IntEnum):
    def __new__(cls, value: int, code: str):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.code = code
        return obj

    @classmethod
    def from_str(cls, s: str):
        for member in cls:
            if member.code == s:
                return member
        raise ValueError(f"{s!r} is not a valid {cls.__name__}")

    def to_str(self) -> str:
        return self.code


class MeterType(CodeIntEnum):
    IAMB = (0, "Я")
    TROCHEE = (1, "Х")
    DACTYL = (2, "Д")
    ANAPEST = (3, "Ан")
    AMPHIBRACH = (4, "Аф")
    DOLNIK = (5, "Дк")
    TAKTOVIK = (6, "Тк")
    AKSTENTNIK = (7, "Ак")
    LOGAED = (8, "Л")
    HEXAMETER = (9, "Гек")
    PAEON = (10, "Пен")
    SYLLABIC = (11, "С")


class Clausula(CodeIntEnum):
    MASCULINE = (0, "м")
    FEMININE = (1, "ж")
    DACTYLIC = (2, "д")
    HYPERDACTYLIC = (3, "г")


class Meter(BaseModel):
    meter: MeterType
    feet: int
    clausula: Clausula
    unstable: bool = Field(default=False)

    def encode(self):
        return [self.meter, self.feet, self.clausula, self.unstable]

    @classmethod
    def decode(cls, data):
        meter, feet, clausula, unstable = data
        return cls(
            meter=meter,
            feet=feet,
            clausula=clausula,
            unstable=unstable,
        )


class Line(BaseModel):
    # строка может содержать несколько метров: например, в случае цезурного разделения строки
    meters: list[Meter]
    # слог, после которого располагается цезура. -1, если цезура отсутствует
    caesura: int
    syllable_masks: SyllableMasks

    def encode(self):
        return [
            self.caesura,
            self.syllable_masks.encode(),
            [m.encode() for m in self.meters],
        ]

    @classmethod
    def decode(cls, data):
        caesura, masks_data, meters_data = data

        return cls(
            caesura=caesura,
            syllable_masks=SyllableMasks.decode(masks_data),
            meters=[Meter.decode(m) for m in meters_data],
        )


class OutputPoem(BaseModel):
    path: str
    lines: list[Line]

    def encode(self):
        return [
            self.path,
            [line.encode() for line in self.lines],
        ]

    @classmethod
    def decode(cls, data):
        path, lines_data = data

        return cls(
            path=path,
            lines=[Line.decode(line) for line in lines_data],
        )
