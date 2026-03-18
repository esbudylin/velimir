from stressrnn import StressRNN

stress_rnn = StressRNN()

accent_mark = "+"

stress_mark_ord = 768


def extract_accent_mask(text: str, accent_mark="+") -> list[bool]:
    result = []

    def is_accent_mark(char):
        return char and ord(char) in [stress_mark_ord, ord(accent_mark)]

    for i, char in enumerate(text):
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if is_vowel(char):
            if is_accent_mark(next_char):
                result.append(True)
            else:
                result.append(False)

    return result


def is_vowel(char):
    vowels = "аеиоуыэюяёАЕИОУЫЭЮЯЁ"

    return char in vowels


def vowel_count(word):
    return sum(map(is_vowel, word))


def remove_accent_marks(text: str) -> str:
    return "".join(c for c in text if ord(c) != stress_mark_ord and c != accent_mark)


def extract_neuro_accents(line) -> list[list[bool]]:
    words = stress_rnn.put_stress(line, accent_mark, use_batch_mode=True).split()

    res = []
    for word in words:
        if mask := extract_accent_mask(word):
            res.append(mask)

    return res
