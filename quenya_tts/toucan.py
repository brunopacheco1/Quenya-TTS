"""
Glue between Quenya and an unmodified IMS-Toucan checkout (the toucan/ git submodule).

Toucan has no Quenya language. Instead of adding one to its code, Quenya text is converted to IPA with
quenya_g2p and handed to Toucan as phonemes, which Toucan accepts for both inference and training.
Quenya borrows the Finnish language embedding, because the pretrained model has none for Quenya.
"""

import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOUCAN_DIR = os.environ.get("TOUCAN_DIR", os.path.join(REPO_DIR, "toucan"))
MODEL_DIR = os.path.join(REPO_DIR, "Models")
CORPORA_DIR = os.path.join(REPO_DIR, "Corpora")

# ISO 639-3 code whose language embedding Quenya borrows
PROXY_LANGUAGE = "fin"


def setup_toucan(change_dir=False):
    """
    Makes Toucan's modules importable and points its storage at this repository's Models/ and Corpora/.

    Toucan imports its modules by top-level name (Preprocessing, Modules, Utility, ...) and reads a few
    files by paths relative to its own folder, so training also needs change_dir=True. Call this before
    importing anything from Toucan, and turn user paths into absolute paths before changing directory.
    """
    if not os.path.exists(os.path.join(TOUCAN_DIR, "Preprocessing", "TextFrontend.py")):
        sys.exit(f"IMS-Toucan not found in {TOUCAN_DIR}. Run: git submodule update --init")
    if TOUCAN_DIR not in sys.path:
        sys.path.insert(0, TOUCAN_DIR)
    import Utility.storage_config as storage_config
    storage_config.MODEL_DIR = MODEL_DIR + os.sep
    storage_config.PREPROCESSING_DIR = CORPORA_DIR + os.sep
    if change_dir:
        os.chdir(TOUCAN_DIR)


class QuenyaPhonemizer:
    """Turns Quenya text into the phoneme strings Toucan expects as input (with its pause and end markers)."""

    def __init__(self, device="cpu"):
        setup_toucan()
        from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
        self.frontend = ArticulatoryCombinedTextFrontend(language=PROXY_LANGUAGE, device=device, silent=True)

    def __call__(self, text):
        from quenya_g2p import to_ipa
        return self.frontend.postprocess_phoneme_string(to_ipa(text),
                                                        for_feature_extraction=True,
                                                        include_eos_symbol=True,
                                                        for_plot_labels=False)
