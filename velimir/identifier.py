import logging
import itertools
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from . import accentuator
from . import parsers
from .domain_models import Clausula, Meter, MeterType, SyllableDistances
from .io import load_models


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


def detect_poetic_accents(model, device, lines: list[str]):
    xs = []
    for line in lines:
        line_with_linguistic_accents = accentuator.accent_line(line)

        xs.append(
            torch.stack(
                [
                    torch.tensor(
                        line_with_linguistic_accents,
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        parsers.extract_word_ending_mask(line),
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
        )

    x = pad_sequence(xs, batch_first=True, padding_value=-1).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()

    mask = (x != -1).all(dim=2)
    return pred.masked_fill(~mask, -1).unsqueeze(-1)


def accent_predictions_to_distances(pred):
    results = []

    pred = pred.squeeze(-1)

    for seq in pred:
        valid = seq[seq != -1]
        accent_pred = valid.bool().cpu().tolist()

        distances = SyllableDistances(accent_pred).to_array()
        results.append(distances)

    return results


def detect_meter(model, device, accent_pred):
    accent_pred = accent_pred.to(device)

    syllable_distances = torch.tensor(
        accent_predictions_to_distances(accent_pred),
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred = model(accent_pred, syllable_distances)

    pred_meter = pred[:, :3]
    pred_caesura = pred[:, 3:5]
    pred_unstable = pred[:, 5]

    return pred_meter, pred_caesura, pred_unstable


def extract_meter_accent_mask(
    meter_position: int,
    caesuras: list[int],
    line_accent_mask: list[bool],
) -> list[bool]:
    if not caesuras:
        return line_accent_mask

    match meter_position:
        case 0:
            return line_accent_mask[: caesuras[0]]
        case 1:
            return line_accent_mask[caesuras[0] : caesuras[1]]
        case 2:
            return line_accent_mask[caesuras[1] : caesuras[2]]


def extract_clausula(meter_accent_mask: list[bool]) -> Clausula:
    last_syllables_without_accent = itertools.takewhile(
        lambda n: not n,
        reversed(meter_accent_mask),
    )
    return Clausula(len(list(last_syllables_without_accent)))


def process_line(meta, pmask) -> ProcessedLine:
    meter_codes, caesura_positions, unstable = meta
    meter_types = []
    line_meters = []

    for code in meter_codes:
        meter_types.append(MeterType(code))

    for i, meter_type in enumerate(meter_types):
        meter_mask = extract_meter_accent_mask(i, caesura_positions, pmask)

        line_meters.append(
            Meter(
                meter=meter_type,
                feet=len([i for i in meter_mask if i]),
                clausula=extract_clausula(meter_mask),
                unstable=unstable,
            )
        )

    return ProcessedLine(
        caesura=caesura_positions,
        meters=line_meters,
        poetic_accent_mask=pmask,
    )


def process_lines(lines: list[str]) -> list[ProcessedLine | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model = load_model(AccentModel, ACCENT_MODEL, device)
    meter_model = load_model(MeterModel, METER_MODEL, device)

    poetic_accent_masks = detect_poetic_accents(
        accent_model,
        device,
        lines,
    )
    meter_preds, caesura_preds, unstable_preds = detect_meter(
        meter_model,
        device,
        poetic_accent_masks,
    )

    def filter_padding(li):
        return list(filter(lambda n: n != -1, li))

    meters_list = []
    for i in range(len(lines)):
        meter_codes = filter_padding(meter_preds[i].round().int().tolist())
        caesura_positions = filter_padding(caesura_preds[i].round().int().tolist())
        unstable = (torch.sigmoid(unstable_preds[i]) > 0.5).item()

        meters_list.append((meter_codes, caesura_positions, unstable))

    poetic_accent_masks_list = []
    for mask in poetic_accent_masks:
        valid_mask = mask[mask != -1]
        poetic_accent_masks_list.append(valid_mask.cpu().numpy().tolist())

    res = []
    for i, (meta, pmask) in enumerate(zip(meters_list, poetic_accent_masks_list)):
        try:
            pl = process_line(meta, pmask)
            res.append(pl)
        except Exception as e:
            logging.error("Failed to process line: %s", lines[i])
            logging.exception(e)
            res.append(None)

    return res
