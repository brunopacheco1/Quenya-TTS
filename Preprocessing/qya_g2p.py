"""
Quenya grapheme-to-phoneme conversion.

Target pronunciation: Third Age Quenya as described in Appendix E of
The Lord of the Rings. The rules are Quenya-only: Sindarin names in mixed
text (Elbereth, Thorondor, Lúthien) get the Quenya th -> [s] rule, although
Sindarin th is [θ], so they should not be used to judge the output.

- Vowels: short a e i o u are [a ɛ i ɔ u]; long á é í ó ú are [aː eː iː oː uː]
  (Appendix E: long é and ó are tenser and closer than the short vowels).
  Circumflex and macron also mark long vowels. A dieresis (ë, ä, ...) only
  marks a vowel that is pronounced on its own; it never ends a diphthong.
- Diphthongs: ai, au, eu, oi, ui (falling), and iu, which Appendix E says was
  usually a rising diphthong [ju] in the Third Age.
- c and k are [k], qu (and Tolkien's older spelling q) is [kw], x is [ks],
  g is [ɡ], y is [j], r is trilled. A stray j is read like y.
- h is [h], but [ç] in hy, [ʍ] in hw, and before t (also before ty) it is [ç]
  after e/i and [x] elsewhere (ht as in German echt, acht).
- hl and hr are [l] and [r]: Appendix E says hl "was in the Third Age usually
  pronounced as l" and refers hr to the same note ("Cf. L"); Eldamo reads this
  as both being voiced in Third Age speech.
- th and þ are [s] (the Exiles' pronunciation of þ: Appendix E, Isil).
- z, which occurs only in archaic or Vanyarin spellings, is read as [r], the
  Third Age (Noldorin) development (áze = áre).
- ty, ny, ly are the palatals [c ɲ ʎ]; other consonant + y is consonant + [j].
  Appendix E's "t in English tune" fits either [c] or [tj]; Eldamo says the
  choice barely matters.
- n is [ŋ] before c/k/qu/g; ng is [ŋɡ], final ng is [ŋ]; ñ is [ŋ]. Initial ñ and
  ng are [n] (Third Age: Ñoldor, ngwalmë are said Noldor, nwalmë).
- w is read as written, [w] (Appendix E: "W has the sound of English w"),
  except at the start of a word, where it is [v] (the Third Age change of
  wilya to vilya); medially (Elwë, Aiwendil, ngw) it stays [w].
- Double consonants are written as two phones (ll -> ll).

Stress: words of two syllables are stressed on the first. Longer words are
stressed on the second-to-last syllable when it is heavy (long vowel,
diphthong, or followed by two or more consonants), otherwise on the
third-to-last. Appendix E does not say how the digraphs count; following
Eldamo ("digraphs like ty and qu are treated as clusters for placement of
stress"), qu, x, ng and any consonant + y or w (ty ny ly ry hy hw) count as
two consonants, while hl, hr and th count as one. Only primary stress is
marked, directly before the stressed vowel, because the text frontend applies
it to the next phone; a long vowel outside the stressed syllable (the ú of
únótimë) would carry secondary stress, which is not marked.

Choices made where the sources disagree and Appendix E does not settle the
matter: archaic or Vanyarin spellings (initial w, z, þ, initial ñ) are read
with their Third Age sounds rather than as written; plain h between vowels is
[h] (Appendix E: "h in house, behold"), not the older spirant [x]; ty, ny, ly
are the single palatal phones rather than consonant + [j]; and the digraphs
weigh for stress as described above.

Text handling: words are stressed independently; the article i and the
particles ar, ve, mi are left unstressed. Compounds joined by a hyphen or a
raised dot (Ambar-metta, ohta·caro) are read as separate words, an apostrophe
marks elision (tenn' = tenn), and line breaks, dashes and parentheses become
pauses like a comma.
"""

import re
import unicodedata

SHORT_VOWELS = {'a': 'a', 'e': 'ɛ', 'i': 'i', 'o': 'ɔ', 'u': 'u'}
LONG_VOWELS = {'a': 'aː', 'e': 'eː', 'i': 'iː', 'o': 'oː', 'u': 'uː'}
DIPHTHONGS = {'ai': 'ai', 'au': 'au', 'eu': 'ɛu', 'oi': 'ɔi', 'ui': 'ui', 'iu': 'ju'}

# Small function words that stay unstressed: the article, "and", "like", "in".
UNSTRESSED_WORDS = {'i', 'ar', 've', 'mi'}

STRESS = 'ˈ'

# Accented letter -> (base vowel, long, separate). "separate" marks a dieresis.
_VOWEL_LETTERS = {}
for _base in 'aeiou':
    _VOWEL_LETTERS[_base] = (_base, False, False)
for _base, _acute, _circ, _macron, _dieresis in zip('aeiou', 'áéíóú', 'âêîôû', 'āēīōū', 'äëïöü'):
    _VOWEL_LETTERS[_acute] = (_base, True, False)
    _VOWEL_LETTERS[_circ] = (_base, True, False)
    _VOWEL_LETTERS[_macron] = (_base, True, False)
    _VOWEL_LETTERS[_dieresis] = (_base, False, True)

# Consonant units, longest first: spelling -> (phones, weight). The weight is
# how many consonants the unit counts as for syllable weight.
_CONSONANT_UNITS = [
    ('qu', 'kw', 2),
    ('kw', 'kw', 2),
    ('q', 'kw', 2),
    ('hw', 'ʍ', 2),
    ('hy', 'ç', 2),
    ('hl', 'l', 1),
    ('hr', 'r', 1),
    ('ty', 'c', 2),
    ('ny', 'ɲ', 2),
    ('ly', 'ʎ', 2),
    ('th', 's', 1),
    ('ng', 'ŋɡ', 2),
    ('x', 'ks', 2),
    ('c', 'k', 1),
    ('k', 'k', 1),
    ('g', 'ɡ', 1),
    ('y', 'j', 1),
    ('þ', 's', 1),
    ('ñ', 'ŋ', 1),
    ('z', 'r', 1),
]
_SIMPLE_CONSONANTS = set('bdfhjlmnprstvw')

PUNCTUATION_MARKS = ';:,.!?¡¿…"“”~/。【】、‥،؟؛'
# Marks that the text frontend does not know; they become a pause like a comma.
_PAUSE_MARKS = ('\n', '–', '—', '(', ')', '«', '»')
# Only the letters Quenya spelling uses; anything else separates words.
_WORD_CHARS = "a-zA-ZáéíóúâêîôûäëïöüāēīōūÁÉÍÓÚÂÊÎÔÛÄËÏÖÜĀĒĪŌŪþÞñÑ"
_TOKEN_RE = re.compile(r"[{w}]+(?:['’][{w}]*)*|[{p}]+".format(w=_WORD_CHARS, p=re.escape(PUNCTUATION_MARKS)))


def _tokenize_word(word):
    """
    Splits a lowercase word into segments. A vowel segment is
    ('V', phones, heavy); a consonant segment is ('C', phones, weight).
    """
    segments = []
    i = 0
    while i < len(word):
        char = word[i]
        if char in _VOWEL_LETTERS:
            base, long, _ = _VOWEL_LETTERS[char]
            if not long and i + 1 < len(word) and word[i + 1] in _VOWEL_LETTERS:
                next_base, next_long, next_separate = _VOWEL_LETTERS[word[i + 1]]
                pair = base + next_base
                if pair in DIPHTHONGS and not next_long and not next_separate:
                    segments.append(('V', DIPHTHONGS[pair], True))
                    i += 2
                    continue
            segments.append(('V', LONG_VOWELS[base] if long else SHORT_VOWELS[base], long))
            i += 1
            continue
        for spelling, phones, weight in _CONSONANT_UNITS:
            if word.startswith(spelling, i):
                segments.append(('C', phones, weight))
                i += len(spelling)
                break
        else:
            if char in _SIMPLE_CONSONANTS:
                segments.append(('C', char, 1))
            i += 1  # anything else (digits, stray symbols) is dropped
    return segments


def _apply_context_rules(segments):
    """
    Rules that depend on neighbouring segments: ht, nasal assimilation,
    initial ñ and ng, w, final ng.
    """
    result = []
    for index, (kind, phones, weight) in enumerate(segments):
        if kind == 'C':
            previous = segments[index - 1] if index > 0 else None
            following = segments[index + 1] if index + 1 < len(segments) else None
            if phones == 'h' and previous is not None and following is not None and following[1][:1] in ('t', 'c'):
                phones = 'ç' if previous[0] == 'V' and previous[1].rstrip('ː')[-1] in 'ɛei' else 'x'
            elif phones == 'n' and following is not None and following[1][:1] in ('k', 'ɡ'):
                phones = 'ŋ'
            elif phones == 'ŋɡ' and following is None:
                phones = 'ŋ'
            elif phones in ('ŋ', 'ŋɡ') and previous is None:
                phones = 'n'
            elif phones == 'w' and previous is None:
                phones = 'v'
        result.append((kind, phones, weight))
    return result


def _stressed_vowel_index(segments):
    vowel_positions = [i for i, segment in enumerate(segments) if segment[0] == 'V']
    if not vowel_positions:
        return None
    if len(vowel_positions) <= 2:
        return vowel_positions[0]
    penultimate = vowel_positions[-2]
    heavy = segments[penultimate][2]
    if not heavy:
        consonants = sum(weight for _, _, weight in segments[penultimate + 1:vowel_positions[-1]])
        heavy = consonants >= 2
    return penultimate if heavy else vowel_positions[-3]


def quenya_word_to_ipa(word, stressed=True):
    """
    Converts one Quenya word (no spaces or punctuation) to IPA with stress.
    """
    word = unicodedata.normalize('NFC', word).lower()
    word = word.replace("'", "").replace("’", "")
    segments = _apply_context_rules(_tokenize_word(word))
    stress_index = _stressed_vowel_index(segments) if stressed else None
    ipa = ''
    for index, (_, phones, _) in enumerate(segments):
        if index == stress_index:
            # the stress goes on the vowel itself, after the glide of a rising diphthong
            if phones.startswith('j'):
                phones = 'j' + STRESS + phones[1:]
            else:
                phones = STRESS + phones
        ipa += phones
    return ipa


def quenya_to_ipa(word):
    # kept for backwards compatibility with older callers
    return quenya_word_to_ipa(word)


def process_complete_sentence(text):
    """
    Converts a Quenya sentence to IPA. Words are separated by spaces and
    punctuation is attached to the preceding word. Compounds joined by a
    hyphen or a raised dot are read as separate words.
    """
    text = unicodedata.normalize('NFC', text)
    for mark in _PAUSE_MARKS:
        text = text.replace(mark, ', ')
    text = re.sub(r'(?:,\s*){2,}', ', ', text)
    for joiner in ('-', '‐', '·'):
        text = text.replace(joiner, ' ')
    processed_parts = []
    for part in _TOKEN_RE.findall(text.strip()):
        if part[0] in PUNCTUATION_MARKS:
            processed_parts.append(part)
        elif any(char in _VOWEL_LETTERS for char in part.lower()):
            stressed = part.lower().strip("'’") not in UNSTRESSED_WORDS
            processed_parts.append(quenya_word_to_ipa(part, stressed=stressed))
        # a token without a vowel (a stray q, an initial) is not a Quenya word and is skipped

    processed_text = ''
    for i, part in enumerate(processed_parts):
        processed_text += part
        if i + 1 < len(processed_parts) and processed_parts[i + 1][0] not in PUNCTUATION_MARKS:
            processed_text += ' '
    return processed_text.strip(', ')
