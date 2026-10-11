"""Tests for the text and alignment helpers of quenya_tts.compare (they need no audio packages)."""

import unittest

from quenya_tts.compare import align
from quenya_tts.compare import find_lines
from quenya_tts.compare import ipa_to_phones
from quenya_tts.compare import phone_error_rate


class PhonesTest(unittest.TestCase):

    def test_drops_stress_pauses_and_length(self):
        self.assertEqual(ipa_to_phones("ˈɛlɛn sˈiːla, lˈuːmɛnn"),
                         ["ɛ", "l", "ɛ", "n", "s", "i", "l", "a", "l", "u", "m", "ɛ", "n", "n"])

    def test_keeps_length_on_request(self):
        self.assertEqual(ipa_to_phones("siːla", keep_length=True), ["s", "iː", "l", "a"])

    def test_unifies_spelling_variants(self):
        self.assertEqual(ipa_to_phones("ɾ g ɹ"), ["r", "ɡ", "r"])

    def test_diacritics_stay_with_their_phone(self):
        self.assertEqual(ipa_to_phones("tʰa çe"), ["tʰ", "a", "ç", "e"])


class AlignTest(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(phone_error_rate(list("abc"), list("abc")), (0.0, []))

    def test_substitution_insertion_deletion(self):
        distance, pairs = align(list("abcd"), list("axcde"))
        self.assertEqual(distance, 2)
        self.assertIn(("b", "x"), pairs)
        self.assertIn((None, "e"), pairs)
        per, diffs = phone_error_rate(list("ab"), list("a"))
        self.assertEqual((per, diffs), (0.5, [("b", None)]))


class FindLinesTest(unittest.TestCase):

    def test_cuts_at_longest_pauses(self):
        # speech stretches with short pauses inside lines and long pauses between them
        intervals = [(0, 10), (12, 20), (40, 50), (52, 60), (90, 100)]
        self.assertEqual(find_lines(intervals, 3), [(0, 20), (40, 60), (90, 100)])

    def test_too_few_stretches(self):
        with self.assertRaises(ValueError):
            find_lines([(0, 10)], 2)


if __name__ == "__main__":
    unittest.main()
