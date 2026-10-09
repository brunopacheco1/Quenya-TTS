"""
Tests for the Quenya grapheme-to-phoneme rules (Third Age pronunciation,
Appendix E). Run from the repository root with:

    python -m unittest discover tests
"""

import re
import unittest

from Preprocessing.qya_g2p import process_complete_sentence
from Preprocessing.qya_g2p import quenya_word_to_ipa


class StressTest(unittest.TestCase):

    def check(self, cases):
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(quenya_word_to_ipa(word), expected)

    def test_two_syllables_stress_the_first(self):
        self.check({
            "Elen":  "ˈɛlɛn",
            "heru":  "hˈɛru",
            "lassi": "lˈassi",
            "Ainur": "ˈainur",
        })

    def test_heavy_penultimate_is_stressed(self):
        self.check({
            "Valinórë":   "valinˈoːrɛ",   # long vowel
            "Endorenna":  "ɛndɔrˈɛnna",   # double consonant
            "omentielvo": "ɔmɛntiˈɛlvɔ",  # consonant cluster
            "atarinya":   "atarˈiɲa",     # ny counts as two consonants
            "quetelyë":   "kwɛtˈɛʎɛ",     # ly counts as two consonants
            "Oiolossë":   "ɔiɔlˈɔssɛ",
        })

    def test_light_penultimate_moves_stress_back(self):
        self.check({
            "utúlien":  "utˈuːliɛn",
            "ancalima": "aŋkˈalima",
            "Eldalië":  "ɛldˈaliɛ",
            "Ilúvatar": "ilˈuːvatar",
            "Eressëa":  "ɛrˈɛssɛa",
            "laurië":   "lˈauriɛ",
        })


class SoundTest(unittest.TestCase):

    def check(self, cases):
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(quenya_word_to_ipa(word), expected)

    def test_vowels(self):
        self.check({
            "Namárië": "namˈaːriɛ",   # ë after i is its own syllable
            "Fëanáro": "fɛanˈaːrɔ",
            "yéni":    "jˈeːni",
            "rámar":   "rˈaːmar",
        })

    def test_diphthongs(self):
        self.check({
            "aiya":  "ˈaija",
            "Ainulindalë": "ainulˈindalɛ",
            "tiuca": "tjˈuka",   # iu is a rising diphthong in the Third Age
        })

    def test_h_combinations(self):
        self.check({
            "hwesta":  "ʍˈɛsta",
            "hyarmen": "çˈarmɛn",
            "hlócë":   "lˈoːkɛ",   # hl is l in the Third Age
            "hrívë":   "hrˈiːvɛ",
            "ohtar":   "ˈɔxtar",
            "tehtar":  "tˈɛçtar",
        })

    def test_nasals(self):
        self.check({
            "ingor":    "ˈiŋɡɔr",
            "Ñoldor":   "nˈɔldɔr",   # initial ñ is n in the Third Age
            "ngwalmë":  "nwˈalmɛ",
            "Angamando": "aŋɡamˈandɔ",
        })

    def test_palatals_and_w(self):
        self.check({
            "tyelpë": "cˈɛlpɛ",
            "Elwë":   "ˈɛlwɛ",
            "wingë":  "vˈiŋɡɛ",   # initial w is v in the Third Age
        })

    def test_th_and_x(self):
        self.check({
            "þúlë":  "sˈuːlɛ",
            "Helcaraxë": "hɛlkarˈaksɛ",
        })


class SentenceTest(unittest.TestCase):

    def test_namarie_line(self):
        self.assertEqual(
            process_complete_sentence("Ai! laurië lantar lassi súrinen, yéni únótimë ve rámar aldaron!"),
            "ˈai! lˈauriɛ lˈantar lˈassi sˈuːrinɛn, jˈeːni uːnˈoːtimɛ vɛ rˈaːmar ˈaldarɔn!")

    def test_hyphen_apostrophe_and_function_words(self):
        self.assertEqual(
            process_complete_sentence("Sinome maruvan ar Hildinyar, tenn’ Ambar-metta!"),
            "sˈinɔmɛ mˈaruvan ar hildˈiɲar, tˈɛnn ˈambar mˈɛtta!")

    def test_every_word_has_one_stress(self):
        text = "Elen síla lúmenn omentielvo. Ma quetelyë Quenya?"
        for word in re.findall(r"[^\s,.!?]+", process_complete_sentence(text)):
            with self.subTest(word=word):
                self.assertEqual(word.count("ˈ"), 1)

    def test_output_only_uses_known_phones(self):
        with open("Preprocessing/articulatory_features.py", encoding="utf8") as f:
            known = set(re.findall(r"^        '([^']+)': \{", f.read(), re.M))
        known |= {"ː", "ˈ", ",", ";", ":"}
        text = ("Ai! laurië lantar lassi súrinen, yéni únótimë ve rámar aldaron! "
                "hwesta hyarmen hlócë hrívë tehtar ohtar ingor Ñoldor tyelpë þúlë Helcaraxë tiuca")
        self.assertEqual(set(process_complete_sentence(text)) - known, set())


if __name__ == "__main__":
    unittest.main()
