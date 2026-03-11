"""
MIT License

Copyright (c) 2021 yuliya1324

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
from collections import defaultdict

from stressrnn import StressRNN

stress_rnn = StressRNN()
accent_mark = "'"


def accent_line(line):
    words = line.split()
    words_rule = accent_line_rules(line).split()
    words_nacc = stress_rnn.put_stress(line, accent_mark, use_batch_mode=True).split()

    res = []

    for j, word in enumerate(words):
        if is_word_without_accent(word):
            res.append(word)
        elif should_use_neuro_accent(word, words_rule[j]):
            res.append(words_nacc[j])
        else:
            res.append(words_rule[j])

    return " ".join(res)


def is_word_without_accent(word):
    """
    Слово остается без ударения, если
    1) в нем нет гласных,
    2) оно односложное (оно содержит одну гласную)
    3) в нем есть буква ё,
    4) или оно относится к предлогам, в которых больше одной гласной, но как правило они стоят в безударной позиции ( обо, недо, изо );
    """
    non_str = ["обо", "изо", "подо", "нибудь"]

    vowels = re.findall("[аеиоуыэюяёАЕИОУЫЭЮЯЁ]", word)

    not_enough_vowels = not vowels or len(vowels) < 2
    has_yo = re.findall("[ёЁ]", word)

    return not_enough_vowels or has_yo or word in non_str


def should_use_neuro_accent(word, word_by_dict):
    """
    Слово берется из строки, размеченной нейросетевым
    акцентуатором, если

    1) словарный акцентуатор поставил в этом слове ударение в двух
    местах, или не поставил совсем и при этом в нем нет буквы ё,

    2) словарный акцентуатор поставил и ударение, и
    букву ё (кроме слов через дефис: например, тёмно-си+ний )
    """
    dictionary_accents = re.findall(accent_mark, word_by_dict)

    multiple_dictionary_accents = len(dictionary_accents) > 1
    has_dictionary_accent = accent_mark in word_by_dict

    has_yo = re.findall("[ёЁ]", word_by_dict)

    return (
        (not has_dictionary_accent and not has_yo)
        or multiple_dictionary_accents
        or (has_yo and has_dictionary_accent)
    )


def read_dict(filename, dic):
    with open(filename, encoding="cp1251") as file_read:
        for line in file_read:
            if line.split():
                word, acc = line.split()
            dic[re.sub(r"\(.+$", "", word)] += rf"\x1{word}={acc}"
    return dic


def normalize(s):
    if re.match("^[А-Я]", s):
        caps = True
    else:
        caps = False
    return s.lower(), caps


def accentw(word):
    voc = "([аеиоуыэюяёАЕИОУЫЭЮЯЁ])"
    if (
        not re.match("[А-я]", word)
        or not re.search(voc, word)
        or re.search("[ёЁ]", word)
    ):
        return word
    key, capitalized = normalize(word)
    vals = None
    for i in range(len(key), -1, -1):
        val = di[key[0:i]]
        if not val:
            continue
        ar = val.split(r"\x1")
        for v in ar:
            if v:
                regex, acc = v.split("=")
                if re.match("^" + regex + "$", key):
                    if not capitalized and "!" in acc:
                        continue
                    vals = acc
                    break
        if vals:
            break
    if not vals or (not capitalized and re.match("!", vals)):
        return word
    vals = set(re.split("[,;]", vals.replace("\n", "")))
    chars = re.sub(voc, r"\1|", word).split("|")
    for val in vals:
        pos, acc = re.findall(r"(\d+)(.*)", val)[0]
        if acc == "" or acc == "!":
            acc = accent_mark
        pos = int(pos)
        if pos > 0:
            chars[pos - 1] += acc
    word = re.sub('Е"', "Ё", re.sub('е"', "ё", "".join(chars)))
    return word


def accent_line_rules(line):
    words = re.findall("[А-яЁё-]+", line)
    for word in words:
        new_word = accentw(word)
        if (not re.search(rf"{word}'", line)) and (word != new_word):
            line = re.sub(word, new_word, line)
    return line


di = defaultdict(str)
di = read_dict(os.path.join("data", "accent_dicts", "accent1.dic"), di)
di = read_dict(os.path.join("data", "accent_dicts", "accent.dic"), di)
