"""Tests for the file handling of quenya_tts.prepare_corpus (no audio packages needed)."""

import os
import tempfile
import unittest

from quenya_tts.prepare_corpus import clip_name, pair_files, read_lines, write_transcript


class PairFilesTest(unittest.TestCase):

    def test_pairs_recordings_with_texts_of_the_same_name(self):
        with tempfile.TemporaryDirectory() as audio, tempfile.TemporaryDirectory() as texts:
            for name in ("namarie.mp3", "markirya.wav", "notext.mp3", "cover.jpg"):
                open(os.path.join(audio, name), "w").close()
            for name in ("namarie.txt", "markirya.txt", "noaudio.txt"):
                open(os.path.join(texts, name), "w").close()
            pairs = pair_files(audio, texts)
        self.assertEqual([stem for stem, _, _ in pairs], ["markirya", "namarie"])
        self.assertTrue(pairs[1][1].endswith("namarie.mp3"))
        self.assertTrue(pairs[1][2].endswith("namarie.txt"))


class TranscriptTest(unittest.TestCase):

    def test_round_trip_with_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            text = os.path.join(folder, "namarie.txt")
            with open(text, "w", encoding="utf8") as f:
                f.write("Ai! laurië lantar lassi súrinen,\n\n  yéni únótimë ve rámar aldaron!  \n")
            lines = read_lines(text)
            self.assertEqual(lines, ["Ai! laurië lantar lassi súrinen,", "yéni únótimë ve rámar aldaron!"])

            transcript = os.path.join(folder, "transcript.txt")
            write_transcript(transcript, [(clip_name("namarie", n), line) for n, line in enumerate(lines, 1)])
            with open(transcript, encoding="utf8") as f:
                self.assertEqual(f.read().splitlines(),
                                 ["namarie_01.wav|Ai! laurië lantar lassi súrinen,",
                                  "namarie_02.wav|yéni únótimë ve rámar aldaron!"])


if __name__ == "__main__":
    unittest.main()
