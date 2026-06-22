import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import accentuator, nlp
from .domain_models import Clausula, Meter, MeterClass, MeterType
from .ml_preprocess import (
    MeterClassRegistry,
    break_into_chunks,
)
from .nlp import PartOfSpeech


@dataclass
class ProcessedLine:
    meters: list[Meter]
    caesura: list[int]
    poetic_accents: list[bool]

    def to_str(self):
        meter_repr = "~".join(m.to_str() for m in self.meters)
        accent_repr = self._mask_to_string(self.poetic_accents, self.caesura)

        return f"{meter_repr} {accent_repr}"

    @staticmethod
    def _mask_to_string(mask: list[bool], caesura: list[int]):
        caesura_mark = "|"
        accent_mark = "*"

        def ms(mask):
            return accent_mark.join(map(str, extract_rhythm(mask)))

        match caesura:
            case []:
                return ms(mask)
            case [ca]:
                return caesura_mark.join(map(ms, (mask[:ca], mask[ca:])))
            case [ca, cb]:
                return caesura_mark.join(map(ms, (mask[:ca], mask[ca:cb], mask[cb:])))
            case _:
                raise ValueError("Invalid caesura sequence length")


def extract_rhythm(accent_mask: list[bool]):
    res = []
    accentless_syllables = 0

    for has_accent in accent_mask:
        if has_accent:
            res.append(accentless_syllables)
            accentless_syllables = 0
        else:
            accentless_syllables += 1

    res.append(accentless_syllables)

    return res


def pad_and_stack(arrays, pad_value=-1):
    max_len = max(a.shape[0] for a in arrays)

    result = np.full(
        (len(arrays), max_len, *arrays[0].shape[1:]),
        pad_value,
        dtype=arrays[0].dtype,
    )

    for i, a in enumerate(arrays):
        result[i, : a.shape[0]] = a

    return result


def extract_input_tensors(
    stanza_breaks: list[int],
    accent_masks: list[list[bool]],
    word_ending_masks: list[list[bool]],
    part_of_speech: list[list[PartOfSpeech]],
):
    accent_input = []
    pos_input = []

    stanzas = break_into_chunks(
        list(zip(accent_masks, word_ending_masks, part_of_speech)),
        stanza_breaks,
    )

    for stanza_lines in stanzas:
        for ling_accent_mask, word_ending_mask, pos in stanza_lines:
            accent_input.append(
                np.stack(
                    [
                        np.array(ling_accent_mask, dtype=np.float32),
                        np.array(word_ending_mask, dtype=np.float32),
                    ],
                    axis=1,
                )
            )

            pos_input.append(np.array(pos, dtype=np.int64))

    accent_input_padded = pad_and_stack(accent_input, pad_value=-1)
    pos_input_padded = pad_and_stack(pos_input, pad_value=-1)

    return accent_input_padded, pos_input_padded


def extract_meter_accent_mask(
    meter_position: int,
    total_meters: int,
    caesuras: list[int],
    line_accent_mask: list[bool],
) -> list[bool]:
    if not caesuras or total_meters == 1:
        return line_accent_mask

    match meter_position, caesuras:
        case 0, [ca]:
            return line_accent_mask[:ca]
        case 1, [ca]:
            return line_accent_mask[ca:]
        case 0, [ca, cb]:
            return line_accent_mask[:ca]
        case 1, [ca, cb]:
            return line_accent_mask[ca:cb]
        case 2, [ca, cb]:
            return line_accent_mask[cb:]
        case _:
            raise ValueError("Invalid combination of meters and caesuras")


def extract_clausula(meter_accent_mask: list[bool]) -> Clausula:
    last_syllables_without_accent = itertools.takewhile(
        lambda n: not n,
        reversed(meter_accent_mask),
    )
    return Clausula(len(list(last_syllables_without_accent)))


def anacrusa_by_meter_type(meter: MeterType) -> int:
    match meter:
        case MeterType.TROCHEE | MeterType.DACTYL:
            return 0
        case MeterType.IAMB | MeterType.AMPHIBRACH:
            return 1
        case MeterType.ANAPEST:
            return 2
        case _:
            raise ValueError(f"Unsupported meter for stress offset: {meter}")


def decode_caesura_positions(
    relative_caesuras: tuple[Fraction, ...],
    meter_types: tuple[MeterType, ...],
    poetic_accents: list[bool],
    word_ending_mask: list[bool],
) -> list[int]:
    target_stresses = [round(frac * sum(poetic_accents)) for frac in relative_caesuras]

    clausula_positions = []

    target_idx = 0
    current_stress_idx = 1

    for i, stress in enumerate(poetic_accents):
        if not stress:
            continue

        if target_idx >= len(target_stresses):
            break

        if current_stress_idx == target_stresses[target_idx]:
            clausula_positions.append(i + 1)
            target_idx += 1

        current_stress_idx += 1

    caesura_positions = []

    for i, pos in enumerate(clausula_positions):
        between_stresses = len(
            list(
                itertools.takewhile(
                    lambda a: not a,
                    poetic_accents[pos:],
                )
            )
        )

        if between_stresses == 0:
            caesura_positions.append(pos)
            continue

        try:
            meter = meter_types[i + 1]
            anacrusa = anacrusa_by_meter_type(meter)
            clausula = between_stresses - anacrusa

            caesura = pos + clausula

            if not word_ending_mask[caesura - 1]:
                # расчитанная цезура приходится на середину слова,
                # вероятно в тексте присутствует цезурное сокращение
                # или наращение
                raise ValueError

            caesura_positions.append(caesura)
        except (IndexError, ValueError):
            # используем позицию первого окончания слова в цезурном "интервале" как позицию цезуры
            # применяется к тоническим размерам и строкам с цезурными наращениями/сокращениями
            caesura_positions.append(
                extract_caesura_from_word_endings(
                    pos,
                    between_stresses,
                    word_ending_mask,
                )
            )

    return caesura_positions


def extract_caesura_from_word_endings(
    clausula_pos: int,
    caesura_gap: int,
    word_ending_mask: list[bool],
):
    caesura_gap_end = clausula_pos + caesura_gap

    begins_with_word_ending = word_ending_mask[clausula_pos - 1]
    word_ending_gap = word_ending_mask[clausula_pos:caesura_gap_end]

    if begins_with_word_ending:
        return clausula_pos

    first_word_end_pos = len(
        list(
            itertools.takewhile(
                lambda a: not a,
                word_ending_gap,
            )
        )
    )

    return clausula_pos + first_word_end_pos + 1


def extract_feet(meter_type: MeterType, meter_mask: list[bool]) -> int:
    stress_count = sum(meter_mask)

    # в силлабике размечается количество слогов, а не ударений
    if meter_type == MeterType.SYLLABIC:
        return len(meter_mask)

    if meter_type in (MeterType.DOLNIK, MeterType.TAKTOVIK):
        # отсекаем анакрусу и клаузлу
        rhythm = extract_rhythm(meter_mask)[1:-1]

        # специальный случай: виртуальный икт
        if rhythm and max(rhythm) >= 4:
            return stress_count + 1

    return stress_count


def process_line(
    mc: MeterClass,
    pmask: list[bool],
    caesuras: list[int],
) -> ProcessedLine:
    line_meters = []

    for i, meter_type in enumerate(mc.meter_types):
        meter_mask = extract_meter_accent_mask(
            meter_position=i,
            total_meters=len(mc.meter_types),
            caesuras=caesuras,
            line_accent_mask=pmask,
        )

        line_meters.append(
            Meter(
                meter=meter_type,
                feet=extract_feet(meter_type, meter_mask),
                clausula=extract_clausula(meter_mask),
                unstable=mc.unstable[i],
            )
        )

    return ProcessedLine(
        caesura=caesuras,
        meters=line_meters,
        poetic_accents=pmask,
    )


def process_lines(
    meter_model,
    accent_model,
    lines: list[str],
    stanza_breaks: list[int],
) -> list[ProcessedLine | None]:
    word_ending_masks = [accentuator.extract_word_ending_mask(li) for li in lines]
    ling_accent_masks = [accentuator.accent_line(li) for li in lines]

    gf_expanded = [
        gf.expand(wem) for gf, wem in zip(map(nlp.markup, lines), word_ending_masks)
    ]

    accent_input, pos_input = extract_input_tensors(
        stanza_breaks,
        ling_accent_masks,
        word_ending_masks,
        [gf.part_of_speech for gf in gf_expanded],
    )

    N = len(lines)
    T_max = accent_input.shape[1]
    all_meter_preds = np.full(N, -1, dtype=np.int64)
    all_accent_mask = np.full((N, T_max), -1.0, dtype=np.float32)

    line_indices = list(range(N))
    chunks = list(break_into_chunks(line_indices, stanza_breaks))

    for stanza_lines in chunks:
        indices = np.array(stanza_lines)
        chunk_accent = accent_input[indices]
        chunk_pos = pos_input[indices]

        meter_logits = meter_model(chunk_accent, chunk_pos)
        meter_preds = np.argmax(meter_logits, axis=1)
        all_meter_preds[indices] = meter_preds

        accent_logits = accent_model(chunk_accent, chunk_pos, meter_preds)
        accent_preds = (1.0 / (1.0 + np.exp(-accent_logits)) > 0.5).astype(np.float32)

        syllable_mask = ~(chunk_accent != -1).all(axis=2)
        accent_preds[syllable_mask] = -1

        all_accent_mask[indices] = accent_preds

    meters_list = [MeterClassRegistry.int_to_mc(int(mi)) for mi in all_meter_preds]

    poetic_accents_list = []
    for mask in all_accent_mask:
        valid = mask[mask != -1]
        poetic_accents_list.append(valid.astype(bool).tolist())

    res = []
    for i, (mc, pmask, wmask) in enumerate(
        zip(meters_list, poetic_accents_list, word_ending_masks)
    ):
        try:
            caesuras = decode_caesura_positions(
                mc.caesura,
                mc.meter_types,
                pmask,
                wmask,
            )
            pl = process_line(mc, pmask, caesuras)
            res.append(pl)
        except Exception as e:
            logging.error("Failed to process line: %s", lines[i])
            logging.exception(e)
            res.append(None)

    return res
