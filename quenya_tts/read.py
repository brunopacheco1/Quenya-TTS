"""
Reads Quenya text aloud and saves it as a WAV file.

Without a Quenya checkpoint it uses the pretrained multilingual ToucanTTS model (downloaded from
Hugging Face on first use), which has never heard Quenya: the pronunciation comes from the quenya_g2p
rules, the voice and rhythm from the languages the model was trained on. After fine-tuning, pass
--model Models/ToucanTTS_Quenya/best.pt to use the Quenya model.

Examples, from the repository root:
    python -m quenya_tts.read "Elen síla lúmenn' omentielvo" -o audios/elen_sila.wav
    python -m quenya_tts.read --file chapter.txt -o audios/chapter.wav
    python -m quenya_tts.read "Namárië" --phonemes   # only print the phonemes, no audio
"""

import argparse
import os

from quenya_g2p import to_ipa


FRAME_SECONDS = 384 / 24000  # one spectrogram frame of the ToucanTTS vocoder


def synthesize(tts, phones, speed=1.0, lead_seconds=0.15):
    """
    Speaks a Toucan phoneme string with a real pause before the first sound.

    Toucan always shortens the opening pause to a single frame, so the speech starts at once and the first
    syllable comes out clipped. Here a second pause symbol follows the opening one, the durations are
    predicted once, that second pause is set to lead_seconds, and the sentence is spoken again with those
    durations. With prosody_creativity=0 both runs predict the same pitch and energy.
    """
    if lead_seconds <= 0:
        return tts(phones, input_is_phones=True, duration_scaling_factor=speed, prosody_creativity=0.0)
    phones = "~" + phones if phones.startswith("~") else "~~" + phones
    _, _, durations, _ = tts(phones, input_is_phones=True, duration_scaling_factor=speed, prosody_creativity=0.0,
                             return_everything=True)
    durations = durations.squeeze(0).clone()
    durations[1] = max(int(durations[1]), round(lead_seconds / FRAME_SECONDS))
    return tts(phones, input_is_phones=True, durations=durations, prosody_creativity=0.0)


def load_reader(model=None, speaker=None, device="cpu"):
    """Returns (tts, phonemize): a ToucanTTS interface with the Quenya settings, and the Quenya phonemizer."""
    from quenya_tts.toucan import PROXY_LANGUAGE, QuenyaPhonemizer, setup_toucan
    setup_toucan()
    from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

    phonemize = QuenyaPhonemizer(device=device)
    tts = ToucanTTSInterface(device=device, tts_model_path=model, language=PROXY_LANGUAGE)
    if speaker:
        tts.set_utterance_embedding(speaker)
    return tts, phonemize


def main():
    parser = argparse.ArgumentParser(description="Read Quenya text aloud with IMS-Toucan.")
    parser.add_argument("text", nargs="?", help="Quenya text to read")
    parser.add_argument("--file", help="read the text from this file instead (one sentence or line per line)")
    parser.add_argument("-o", "--output", default="audios/quenya.wav", help="where to save the WAV file")
    parser.add_argument("--model", default=None, help="ToucanTTS checkpoint (.pt); default: the pretrained multilingual model")
    parser.add_argument("--speaker", default=None, help="a WAV file whose voice the model should imitate")
    parser.add_argument("--speed", type=float, default=1.0, help="duration scaling: above 1 is slower, below 1 faster")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--lead", type=float, default=0.15,
                        help="seconds of pause the model speaks before the first sound (Toucan otherwise starts "
                             "speaking at once, which clips the first syllable)")
    parser.add_argument("--padding", type=float, default=0.3,
                        help="seconds of silence at the start and end of the file (many players and Bluetooth "
                             "headphones swallow the first moments of a file, which cuts off the first syllable)")
    parser.add_argument("--phonemes", action="store_true", help="only print the phonemes and exit")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf8") as f:
            lines = [line.strip() for line in f if line.strip()]
    elif args.text:
        lines = [args.text]
    else:
        parser.error("give some text or --file")

    for line in lines:
        print(f"{line}\n  {to_ipa(line)}")
    if args.phonemes:
        return

    output = os.path.abspath(args.output)
    model = os.path.abspath(args.model) if args.model else None
    speaker = os.path.abspath(args.speaker) if args.speaker else None

    tts, phonemize = load_reader(model=model, speaker=speaker, device=args.device)
    import soundfile
    import torch
    from Utility.utils import float2pcm

    # the same loop as ToucanTTSInterface.read_to_file, but with phonemes as input and longer silence at both ends
    sr = 24000
    padding = torch.zeros([int(args.padding * sr)])
    silence = torch.zeros([400])
    wav = padding.clone()
    for line in lines:
        spoken, sr = synthesize(tts, phonemize(line), speed=args.speed, lead_seconds=args.lead)
        wav = torch.cat((wav, torch.tensor(spoken).cpu(), silence), 0)
    wav = torch.cat((wav, padding), 0)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    soundfile.write(file=output, data=float2pcm(wav), samplerate=sr, subtype="PCM_16")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
