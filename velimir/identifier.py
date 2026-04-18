import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch.nn.utils.rnn import pad_sequence
from velimir.domain_models import MeterType

from . import accentuator, parsers
from .domain_models import Clausula, Meter, MeterClass
from .io import load_models
from .ml_loader import (
    MeterClassRegistry,
    break_into_stanzas,
    compute_mean_ling_accents_per_stanza,
)


@dataclass
class ProcessedLine:
    meters: list[Meter]
    caesura: list[int]
    poetic_accent_mask: list[bool]

    def to_str(self):
        meter_repr = "~".join(m.to_str() for m in self.meters)
        accent_repr = self._mask_to_string(self.poetic_accent_mask, self.caesura)

        return f"{meter_repr} {accent_repr}"

    @staticmethod
    def _mask_to_string(mask: list[bool], caesura: list[int]):
        caesura_mark = "|"
        acccent_mark = "*"

        def ms(mask):
            res = ""
            accentless_syllables = 0

            for i, has_accent in enumerate(mask):
                if has_accent:
                    res += str(accentless_syllables)
                    accentless_syllables = 0

                    res += acccent_mark
                else:
                    accentless_syllables += 1

            res += str(accentless_syllables)

            return res

        match caesura:
            case []:
                return ms(mask)
            case [ca]:
                return caesura_mark.join(map(ms, (mask[:ca], mask[ca:])))
            case [ca, cb]:
                return caesura_mark.join(map(ms, (mask[:ca], mask[ca:cb], mask[cb:])))
            case _:
                raise ValueError("Invalid caesura sequence length")


def extract_accent_input(
    stanza_breaks: list[int],
    accent_masks: list[list[bool]],
    word_ending_masks: list[list[bool]],
):
    xs = []

    stanza_stats = compute_mean_ling_accents_per_stanza(
        accent_masks,
        stanza_breaks,
    )
    stanzas = break_into_stanzas(
        list(zip(accent_masks, word_ending_masks)),
        stanza_breaks,
    )

    for current_stanza, stanza_lines in enumerate(stanzas):
        for ling_accent_mask, word_ending_mask in stanza_lines:
            stanza_stat = stanza_stats[current_stanza][: len(ling_accent_mask)]

            xs.append(
                torch.stack(
                    [
                        torch.tensor(
                            stanza_stat,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            ling_accent_mask,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            word_ending_mask,
                            dtype=torch.float32,
                        ),
                    ],
                    dim=1,
                )
            )

    return pad_sequence(xs, batch_first=True, padding_value=-1)


def detect_poetic_accents(model, device, accent_input, meter_pred):
    accent_input = accent_input.to(device)

    with torch.no_grad():
        logits = model(accent_input, meter_pred)
        pred = (torch.sigmoid(logits) > 0.5).float()

    mask = (accent_input != -1).all(dim=2)
    return pred.masked_fill(~mask, -1).unsqueeze(-1)


def detect_meter(model, device, accent_input):
    accent_input = accent_input.to(device)

    with torch.no_grad():
        pred = model(accent_input)

    return torch.argmax(pred, dim=1)


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


def decode_caesura_positions(
    relative_caesuras: tuple[Fraction, ...],
    meter_types: tuple[MeterType, ...],
    poetic_accent_mask: list[bool],
    word_ending_mask: list[bool],
) -> list[int]:
    target_stresses = [
        round(frac * sum(poetic_accent_mask)) for frac in relative_caesuras
    ]

    clausula_positions = []

    target_idx = 0
    current_stress_idx = 1

    for i, stress in enumerate(poetic_accent_mask):
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
                    poetic_accent_mask[pos:],
                )
            )
        )

        if between_stresses == 0:
            caesura_positions.append(pos)
            continue

        try:
            meter = meter_types[i + 1]
            anacrusa = parsers.stress_position_in_foot(meter)
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
                feet=len([i for i in meter_mask if i]),
                clausula=extract_clausula(meter_mask),
                unstable=mc.unstable[i],
            )
        )

    return ProcessedLine(
        caesura=caesuras,
        meters=line_meters,
        poetic_accent_mask=pmask,
    )


def process_lines(
    lines: list[str],
    stanza_breaks: list[int],
) -> list[ProcessedLine | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model, meter_model = load_models(device)

    word_ending_masks = [parsers.extract_word_ending_mask(li) for li in lines]
    ling_accent_masks = [accentuator.accent_line(li) for li in lines]

    accent_input = extract_accent_input(
        stanza_breaks,
        ling_accent_masks,
        word_ending_masks,
    )
    meter_preds = detect_meter(
        meter_model,
        device,
        accent_input,
    )
    poetic_accent_masks = detect_poetic_accents(
        accent_model,
        device,
        accent_input,
        meter_preds,
    )

    def filter_padding(li):
        return list(filter(lambda n: n != -1, li))

    meters_list = []
    for i in range(len(lines)):
        mi = meter_preds[i]
        mc = MeterClassRegistry.int_to_mc(mi)

        meters_list.append(mc)

    poetic_accent_masks_list = []
    for mask in poetic_accent_masks:
        valid_mask = mask[mask != -1]
        poetic_accent_masks_list.append(valid_mask.cpu().numpy().tolist())

    res = []
    for i, (mc, pmask, wmask) in enumerate(
        zip(meters_list, poetic_accent_masks_list, word_ending_masks)
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
