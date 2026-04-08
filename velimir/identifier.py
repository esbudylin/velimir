import itertools
import logging
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from . import accentuator, parsers
from .domain_models import Clausula, Meter, MeterType, MeterClass
from .io import load_models
from .ml_loader import compute_mean_ling_accents_per_stanza, MeterClassRegistry


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


# TODO: remove code duplication with ml_loader
def detect_poetic_accents(
    model,
    device,
    lines: list[str],
    stanza_breaks: list[int],
):
    xs = []
    ling_accent_masks = [accentuator.accent_line(li) for li in lines]

    stanza_stats = compute_mean_ling_accents_per_stanza(
        stanza_breaks,
        ling_accent_masks,
    )
    current_stanza = 0

    for i, (ling_accent_mask, line) in enumerate(zip(ling_accent_masks, lines)):
        if (
            len(stanza_breaks) != current_stanza + 1
            and i == stanza_breaks[current_stanza + 1]
        ):
            current_stanza += 1

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


def detect_meter(model, device, accent_pred):
    accent_pred = accent_pred.to(device)

    with torch.no_grad():
        pred = model(accent_pred)

    return torch.argmax(pred, dim=1)


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


def process_line(mc: MeterClass, pmask: list[bool]) -> ProcessedLine:
    line_meters = []

    caesura_positions = mc.decode_caesura_positions(pmask)

    for i, meter_type in enumerate(mc.meter_types):
        meter_mask = extract_meter_accent_mask(i, caesura_positions, pmask)

        line_meters.append(
            Meter(
                meter=meter_type,
                feet=len([i for i in meter_mask if i]),
                clausula=extract_clausula(meter_mask),
                unstable=mc.unstable[i],
            )
        )

    return ProcessedLine(
        caesura=caesura_positions,
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

    poetic_accent_masks = detect_poetic_accents(
        accent_model,
        device,
        lines,
        stanza_breaks,
    )
    meter_preds = detect_meter(
        meter_model,
        device,
        poetic_accent_masks,
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
    for i, (mc, pmask) in enumerate(zip(meters_list, poetic_accent_masks_list)):
        try:
            pl = process_line(mc, pmask)
            res.append(pl)
        except Exception as e:
            logging.error("Failed to process line: %s", lines[i])
            logging.exception(e)
            res.append(None)

    return res
