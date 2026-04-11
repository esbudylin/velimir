from dataclasses import dataclass, fields
from enum import IntEnum

import bitarray.util as bu
from bitarray import bitarray


@dataclass
class InputPoem:
    author: str
    created: str
    header: str
    formula: str
    meter: str
    clausula: str
    feet: str
    path: str

    @classmethod
    def from_row(cls, d):
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class InputLine:
    meter: str
    text: str


@dataclass(slots=True)
class SyllableMasks:
    linguistic_accent_mask: bitarray
    poetic_accent_mask: bitarray
    last_in_word_mask: bitarray

    def __post_init__(self):
        if not isinstance(self.linguistic_accent_mask, bitarray):
            self.linguistic_accent_mask = bitarray(self.linguistic_accent_mask)

        if not isinstance(self.poetic_accent_mask, bitarray):
            self.poetic_accent_mask = bitarray(self.poetic_accent_mask)

        if not isinstance(self.last_in_word_mask, bitarray):
            self.last_in_word_mask = bitarray(self.last_in_word_mask)

        l_len = len(self.linguistic_accent_mask)
        p_len = len(self.poetic_accent_mask)
        w_len = len(self.last_in_word_mask)

        if not l_len or not p_len or not w_len:
            raise ValueError("Masks are empty")

        if l_len != p_len or p_len != w_len:
            raise ValueError("Masks must have the same length")

    def encode(self):
        return [
            bu.serialize(self.linguistic_accent_mask),
            bu.serialize(self.poetic_accent_mask),
            bu.serialize(self.last_in_word_mask),
        ]

    @classmethod
    def decode(cls, data):
        return cls(
            linguistic_accent_mask=bu.deserialize(data[0]),
            poetic_accent_mask=bu.deserialize(data[1]),
            last_in_word_mask=bu.deserialize(data[2]),
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


# Simlified representation of a line's meter
# used for classification in ML models
@dataclass(frozen=True, slots=True)
class MeterClass:
    meter_types: tuple[MeterType]
    # позиции цезурных разделений относительно количества поэтических ударений в строке
    caesura: tuple[float]
    unstable: tuple[bool]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            meter_types=tuple(MeterType(mt) for mt in data["meter_types"]),
            caesura=tuple(data["caesura"]),
            unstable=tuple(data["unstable"]),
        )


@dataclass(slots=True)
class Line:
    # строка может содержать несколько метров: например, в случае цезурного разделения строки
    meters: list[Meter]
    # позиции слогов, после которых располагается цезура
    caesura: list[int]
    syllable_masks: SyllableMasks

    def to_meterclass(self) -> MeterClass:
        feet = sum(self.syllable_masks.poetic_accent_mask)

        return MeterClass(
            tuple(m.meter for m in self.meters),
            tuple(
                sum(self.syllable_masks.poetic_accent_mask[:c]) / feet
                for c in self.caesura
            ),
            tuple(m.unstable for m in self.meters),
        )

    def length(self):
        # маски - равной длины, здесь можно использовать любую маску
        return len(self.syllable_masks.linguistic_accent_mask)

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
