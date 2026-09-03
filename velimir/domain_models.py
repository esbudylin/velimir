from dataclasses import dataclass, fields
from enum import IntEnum
from fractions import Fraction

import bitarray.util as bu
from bitarray import bitarray


class CodeIntEnum(IntEnum):
    def __init_subclass__(cls):
        cls._code_to_member = {}

    def __new__(cls, value: int, code: str):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.code = code

        cls._code_to_member[code] = obj

        return obj

    @classmethod
    def from_str(cls, s: str):
        try:
            return cls._code_to_member[s]
        except KeyError:
            raise ValueError(f"{s!r} is not a valid {cls.__name__}")

    def to_str(self) -> str:
        return self.code


class MeterType(CodeIntEnum):
    IAMB = 0, "Я"
    TROCHEE = 1, "Х"
    DACTYL = 2, "Д"
    ANAPEST = 3, "Ан"
    AMPHIBRACH = 4, "Аф"
    DOLNIK = 5, "Дк"
    TAKTOVIK = 6, "Тк"
    AKSTENTNIK = 7, "Ак"
    LOGAED = 8, "Л"
    HEXAMETER = 9, "Гек"
    PAEON = 10, "Пен"
    SYLLABIC = 11, "С"


class Clausula(CodeIntEnum):
    MASCULINE = 0, "м"
    FEMININE = 1, "ж"
    DACTYLIC = 2, "д"
    HYPERDACTYLIC = 3, "г"


@dataclass
class InputPoem:
    author: str
    created: str
    header: str
    formula: str
    meter: str
    clausula: str
    feet: str
    rhyme: str
    extra: str
    path: str

    @classmethod
    def from_row(cls, d):
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class InputLine:
    idx: int
    meter: str
    text: str
    rhyme_zone: str  # текст в зоне рифмовки


@dataclass(slots=True)
class SyllableFeatures:
    linguistic_accents: bitarray
    poetic_accents: bitarray
    last_in_word: bitarray

    def __post_init__(self):
        if not isinstance(self.linguistic_accents, bitarray):
            self.linguistic_accents = bitarray(self.linguistic_accents)

        if not isinstance(self.poetic_accents, bitarray):
            self.poetic_accents = bitarray(self.poetic_accents)

        if not isinstance(self.last_in_word, bitarray):
            self.last_in_word = bitarray(self.last_in_word)

        inputs = [
            self.linguistic_accents,
            self.poetic_accents,
            self.last_in_word,
        ]

        lengths = set(map(len, inputs))

        if len(lengths) != 1:
            raise ValueError("Masks must have the same length")

        if 0 in lengths:
            raise ValueError("Masks are empty")

    def encode(self):
        return [
            bu.serialize(self.linguistic_accents),
            bu.serialize(self.poetic_accents),
            bu.serialize(self.last_in_word),
        ]

    @classmethod
    def decode(cls, data):
        return cls(
            linguistic_accents=bu.deserialize(data[0]),
            poetic_accents=bu.deserialize(data[1]),
            last_in_word=bu.deserialize(data[2]),
        )


@dataclass(slots=True)
class Meter:
    meter: MeterType
    feet: int
    clausula: Clausula
    unstable: bool = False  # метр с перебоем

    def to_str(self):
        li = [
            self.meter.to_str(),
            "" if not self.unstable else "*",
            str(self.feet),
            self.clausula.to_str(),
        ]
        return "".join(li)

    def encode(self):
        return [self.meter, self.feet, self.clausula, self.unstable]

    @classmethod
    def decode(cls, data):
        meter, feet, clausula, unstable = data
        return cls(
            meter=MeterType(meter),
            feet=feet,
            clausula=Clausula(clausula),
            unstable=unstable,
        )


# Simplified representation of a line's meter
# used for classification in ML models
@dataclass(frozen=True, slots=True)
class MeterClass:
    meter_types: tuple[MeterType, ...]
    caesura: tuple[Fraction, ...]
    unstable: tuple[bool, ...]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            meter_types=tuple(MeterType(mt) for mt in data["meter_types"]),
            caesura=tuple(map(Fraction, data["caesura"])),
            unstable=tuple(data["unstable"]),
        )


@dataclass(slots=True)
class Line:
    # позиция строки в тексте (с учётом строк, пропущенных при парсинге)
    idx: int
    # строка может содержать несколько метров: например, в случае цезурного разделения строки
    meters: list[Meter]
    # позиции цезурных разделений относительно количества поэтических ударений в строке
    caesura: list[Fraction]
    syllables: SyllableFeatures

    def to_meterclass(self) -> MeterClass:
        return MeterClass(
            tuple(m.meter for m in self.meters),
            tuple(self.caesura),
            tuple(m.unstable for m in self.meters),
        )

    def length(self):
        # маски - равной длины, здесь можно использовать любую маску
        return len(self.syllables.linguistic_accents)

    def encode(self):
        return [
            self.idx,
            [(c.numerator, c.denominator) for c in self.caesura],
            self.syllables.encode(),
            [m.encode() for m in self.meters],
        ]

    @classmethod
    def decode(cls, data):
        idx, caesura, masks_data, meters_data = data

        return cls(
            idx=idx,
            caesura=[Fraction(c[0], c[1]) for c in caesura],
            syllables=SyllableFeatures.decode(masks_data),
            meters=[Meter.decode(m) for m in meters_data],
        )


@dataclass(slots=True)
class Poem:
    path: str
    lines: list[Line]
    # разбивка на строфы: позиция первой строки для каждой строфы
    stanza_breaks: list[int]

    def __post_init__(self):
        if not self.stanza_breaks:
            raise ValueError("Attempted to record a poem without stanza breaks")

        if not self.lines:
            raise ValueError("Attempted to record a poem without lines")

    def encode(self):
        return [
            self.path,
            [line.encode() for line in self.lines],
            self.stanza_breaks,
        ]

    @classmethod
    def decode(cls, data):
        path, lines_data, stanza_breaks = data

        return cls(
            path=path,
            lines=[Line.decode(line) for line in lines_data],
            stanza_breaks=stanza_breaks,
        )
