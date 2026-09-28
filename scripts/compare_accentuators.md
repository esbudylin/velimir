# Accentuator comparison

Results of `scripts/compare_accentuators.py` over all RNC accented (`Ак`)
lines (11,110 lines). Compared models: the project's own accentuator
(dictionaries + StressRNN), `ruaccent`, and `silero-stress`.

`analysed` counts polysyllabic words that carry a stress mark in the corpus;
`correct` counts those where the model's placement agrees. `avg_diff` is the
mean, over lines, of the fraction of analysed words with a wrong stress.

## Without context

| accentuator | lines | words | analysed | correct | word_acc | avg_diff |
|-------------|------:|------:|---------:|--------:|---------:|---------:|
| ours        | 11109 | 58603 |    41275 |   38910 |   0.9427 |   0.0447 |
| ruaccent    | 11109 | 58603 |    41275 |   38597 |   0.9351 |   0.0479 |
| silero      | 11109 | 58602 |    41274 |   39744 |   0.9629 |   0.0289 |

## With neighbouring lines as context (`--context`)

| accentuator | lines | words | analysed | correct | word_acc | avg_diff |
|-------------|------:|------:|---------:|--------:|---------:|---------:|
| ours        | 11109 | 58603 |    41275 |   38910 |   0.9427 |   0.0447 |
| ruaccent    | 11110 | 58605 |    41277 |   37021 |   0.8969 |   0.0753 |
| silero      | 11109 | 58602 |    41274 |   39761 |   0.9633 |   0.0285 |

## Notes

- `silero-stress` is the most accurate on this subset.
- Appending the previous and next lines has no effect on the project's
  accentuator (verified: target masks differ on 0/300 lines), degrades
  `ruaccent` noticeably, and barely affects `silero`.
- `ruaccent`'s neural fallback requires `transformers<5` (pinned in the
  `accent` dependency group).
