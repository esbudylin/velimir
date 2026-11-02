import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def is_word(word):
    if len(word) == 1:
        return word in ["к", "в", "я", "а", "с", "и", "о"]

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        word.replace("и", "й"),
    ]
    return any(map(morph.word_is_known, forms))


def find_acro_variants(inp: str) -> list[list[str]]:
    if len(inp) == 0:
        raise ValueError

    acc = []
    for i in range(len(inp), 0, -1):
        candidate = inp[:i]

        if not is_word(candidate):
            continue

        if i == len(inp):
            acc.append([candidate])
            continue

        solved = find_acro_variants(inp[i:])
        if solved:
            for candidates in solved:
                acc.append([candidate, *candidates])

    return acc


def detect_acro(inp: str) -> list[str]:
    variants = find_acro_variants(inp)

    if variants:
        best = sorted(find_acro_variants(inp), key=len)[0]

        def count_shorts(threshold):
            return len([n for n in best if len(n) < threshold])

        if count_shorts(3) / len(best) > 0.4:
            return []

        if count_shorts(4) / len(best) > 0.5:
            return []

        return best
    else:
        return []
