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


def main():
    parser = argparse.ArgumentParser(description="Read Quenya text aloud with IMS-Toucan.")
    parser.add_argument("text", nargs="?", help="Quenya text to read")
    parser.add_argument("--file", help="read the text from this file instead (one sentence or line per line)")
    parser.add_argument("-o", "--output", default="audios/quenya.wav", help="where to save the WAV file")
    parser.add_argument("--model", default=None, help="ToucanTTS checkpoint (.pt); default: the pretrained multilingual model")
    parser.add_argument("--speaker", default=None, help="a WAV file whose voice the model should imitate")
    parser.add_argument("--speed", type=float, default=1.0, help="duration scaling: above 1 is slower, below 1 faster")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
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

    from quenya_tts.toucan import PROXY_LANGUAGE, QuenyaPhonemizer, setup_toucan
    setup_toucan()
    import soundfile
    import torch
    from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
    from Utility.utils import float2pcm

    phonemize = QuenyaPhonemizer(device=args.device)
    tts = ToucanTTSInterface(device=args.device, tts_model_path=model, language=PROXY_LANGUAGE)
    if speaker:
        tts.set_utterance_embedding(speaker)

    # the same loop as ToucanTTSInterface.read_to_file, but with phonemes as input
    silence = torch.zeros([400])
    wav = silence.clone()
    sr = 24000
    for line in lines:
        spoken, sr = tts(phonemize(line), input_is_phones=True, duration_scaling_factor=args.speed, prosody_creativity=0.0)
        wav = torch.cat((wav, torch.tensor(spoken).cpu(), silence), 0)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    soundfile.write(file=output, data=float2pcm(wav), samplerate=sr, subtype="PCM_16")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
