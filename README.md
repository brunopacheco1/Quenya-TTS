# Quenya-TTS

Text-to-speech for Quenya, built on [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It started as Annie Zhou's master's thesis at the University of Groningen (described below) and is being continued to produce audio for the community book [Ma Quetelyë Quenya?](https://ma-quetelye-quenya.tolkien.lu/).

## How it fits together

Toucan is used as it is, without changes. Quenya text is turned into IPA by our own rules and handed to Toucan as phonemes, which Toucan accepts for reading and for training. Toucan has no Quenya language embedding, so Quenya borrows the Finnish one.

- **[quenya_g2p/](quenya_g2p)** turns Quenya text into IPA. It follows Third Age Quenya as described in Appendix E of The Lord of the Rings; the rules, and the choices made where sources disagree, are documented in [quenya_g2p/rules.py](quenya_g2p/rules.py). It is a small Python package with no dependencies, so other projects can use it too. Tests are in [tests/](tests).
- **[quenya_tts/](quenya_tts)** connects it to Toucan: a reader that turns Quenya text into a WAV file, a training script, and the corpus loaders.
- **[toucan/](toucan)** is IMS-Toucan as a git submodule, pinned to a commit of its `MassiveScaleToucan` branch. Updating Toucan means moving the submodule to a newer commit.

There is no trained Quenya checkpoint yet. The 2024 thesis code, which was a modified copy of an older Toucan, is in this repository's git history before the move to this layout.

## Quick start

Quenya to IPA needs only Python:

```
pip install .            # or: pip install git+https://github.com/brunopacheco1/Quenya-TTS
python -m quenya_g2p "Elen síla lúmenn' omentielvo"
python -m unittest discover tests
```

To hear it, get Toucan and its requirements. Use Python 3.10 or 3.11, because some of Toucan's pinned packages do not install on newer versions. Toucan also needs espeak-ng (`brew install espeak-ng` on macOS, `apt install espeak-ng libportaudio2` on Debian or Ubuntu).

```
git submodule update --init
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m quenya_tts.read "Elen síla lúmenn' omentielvo" -o audios/elen_sila.wav
```

On macOS, if phonemizer cannot find espeak-ng, set `PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib`.

The reader uses the pretrained multilingual Toucan model, downloaded from Hugging Face on first use. That model has never heard Quenya, so the pronunciation follows our rules but the voice and rhythm are generic.

## Comparing with a recording

`quenya_tts.compare` checks our output against a reference recording of the same text, such as a reading of Namárië. For each line it lists what a phone recogniser hears in both recordings next to the IPA from our rules, plots both pitch contours aligned in time, and saves both versions of the line as WAV files. It needs one more package:

```
pip install "transformers>=4.46,<4.50"
python -m quenya_tts.compare path/to/recording.mp3 --text namarie.txt -o comparisons/namarie
```

The text file has one line per spoken line. The recording is cut into lines where the recogniser hears each line's sounds, so lines the reader runs together are still found. Check the cuts with `--segments_only`; if one is wrong, fix the times in `segments.csv` and run again with `--segments comparisons/namarie/segments.csv`. Keep recordings you have no rights to outside the repository.

## Training

Each corpus is a folder with `transcript.txt` (one `file_name|Quenya text` per line) next to its audio files, under `Corpora/Quenya/<name>` (or `$QUENYA_CORPORA_DIR/<name>`):

- `Volunteers`: recordings made for the book by community volunteers under CC-BY or CC0. The public model is trained only on these.
- `Atanquesta`, `Glaemscrafu`: the recordings used in the thesis. Atanquesta is CC BY-NC-SA 4.0, so models trained on these are for experiments only. Pass `--experimental` to include them.

Training needs a CUDA GPU:

```
python -m quenya_tts.train --gpu_id 0
python -m quenya_tts.read "Namárië" --model Models/ToucanTTS_Quenya/best.pt -o audios/namarie.wav
```

## Master's thesis (2024)

This project is a Quenya Text-to-Speech (TTS) system created using [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It is part of a master's thesis project for the MSc in Voice Technology at the [University of Groningen](https://www.rug.nl/). The demo audio can be found [here](https://annie-zhou1997.github.io/QuenyaTTS.github.io/).

This study evaluates the use of articulatory features as inputs for speech synthesis and examines the outcomes after applying transfer learning from models based on more resourced languages. The project utilizes the ToucanTTS system from the University of Stuttgart, which is based on the FastSpeech2 architecture and converts text into articulatory features for speech synthesis. A TTS system for Quenya was developed by fine-tuning three models using a 34-minute Quenya dataset: one for Finnish, one for English, and a multilingual model provided by ToucanTTS. The results showed that the Finnish fine-tuned model produced better speech quality than the English fine-tuned model, while the multilingual fine-tuned model produced the most natural and accurate speech.


### Acknowledgements
I would like to express my deepest gratitude to my supervisor, Ph.D. Candidate Phat Do, and Associate Professor Dr. Matt Coler, for their invaluable guidance throughout this project. My thanks also go to the original creators of the [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) system for providing such a convenient tool. Special thanks to the author of the book [Atanquesta](https://middangeard.org.uk/aglardh/atanquesta), whose open-source recordings were crucial to this project. I am also deeply grateful to [Glaemscrafu](https://glaemscrafu.jrrvf.com/english/index.html), a website rich with resources, particularly the recordings by Benjamin Babut and Bertrand Bellet. Additionally, I extend my gratitude to all the participants in the questionnaire survey and the Quenya enthusiasts who were willing to write detailed listening reports. Finally, I am deeply grateful to the volunteer speaker, Igor Marchenko. Despite your busy schedule leading up to graduation, you provided exceptionally high-quality recordings. You are the best classmate I have ever had.

### Experiment

_The thesis worked in a modified copy of an older Toucan, so the paths below refer to that copy, which is in the git history._

Based on the tutorial from the original repository, I have added Quenya language (qya) support in the `Preprocessing/TextFrontend.py` file and created the Quenya language G2P script, which is also located in the `Preprocessing` directory. The specific steps are as follows:

1. **Adding Quenya Language Support**:
    - Added support for the Quenya language in the `Preprocessing/TextFrontend.py` file.
    - Ensured the new language code (qya) correctly handles the special characters and phonetic rules of the Quenya language.

2. **Creating Quenya G2P Script**:
    - Developed a dedicated Quenya G2P script in the `Preprocessing` directory.
    - The script utilizes knowledge of Quenya phonology to convert text into phonemes and annotate stress.
    - It also considers some obsolete letters from early Quenya variants.

3. **Data Preprocessing**:
    - Manually segmented audio files using Audacity, ensuring each file is between 2 to 13 seconds long with accurate transcription texts.
    - Performed noise reduction on public recordings from Glæmscrafu using voicefixer.

4. **Model Training**:
    - Trained Finnish and English TTS models from scratch using the ToucanTTS system.
    - Used identical training configurations: batch size of 12, learning rate of 1e-3, and 80,000 training steps.
    - After training, generated 10 sentences not present in the training set and tested the word error rate (WER) using OpenAI’s Whisper.

5. **Fine-Tuning**:
    - Integrated the Quenya G2P script into the text front end of ToucanTTS and performed fine-tuning.
    - Fine-tuned the Finnish and English models using a learning rate of 1e-5, batch size of 6, and 6000 training steps.
    - Further fine-tuned using a multilingual pre-trained checkpoint developed with data from 12 languages: English, German, Spanish, Greek, Finnish, French, Russian, Hungarian, Dutch, Polish, Portuguese, and Italian, totaling 389 hours.
    - Averaged and consolidated the final three checkpoints into a single optimized checkpoint for inference.

6. **Evaluation**:
    - Assessed the quality of the synthesized speech using traditional Mean Opinion Score (MOS) evaluations and detailed listening reports.
    - Distributed surveys to Quenya enthusiasts and linguists, and invited two proficient Quenya speakers to identify specific errors in the generated sentences and provide detailed listening reports.
    - In the MOS evaluation, participants were provided with 9 sentences in 4 different versions: real human voice recordings, sentences fine-tuned from a multilingual model, sentences fine-tuned from Finnish, and sentences fine-tuned from English. Participants were not informed which voices were produced by humans and rated each sample on a scale from 1 to 5.
