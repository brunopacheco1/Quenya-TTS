"""
Records a voice for training, one line of the recording script at a time.

Shows each line of a session of recording/script.tsv, records it while you read, and saves it as <id>.wav in
the output folder. Lines already recorded there are skipped, so you can stop at any time and go on later.
After each take you can listen to it, redo it or keep it. See recording/README.md for how to read.

Keys (press Enter after each):
    Enter   start recording; Enter again to stop
    p       play the take back
    r       record the line again
    s       skip the line for now
    q       quit (the take you are on is not saved unless you kept it)

Example, from the repository root:
    python -m quenya_tts.record --script recording/script.tsv --session 1 -o ~/recordings
"""

import argparse
import csv
import os
import queue


def session_lines(script_path, session):
    """Returns [(id, text, gloss)] of one session (1 = ids s01_...), in order."""
    prefix = f"s{session:02d}_"
    with open(script_path, encoding="utf8", newline="") as f:
        return [(row["id"], row["text"], row.get("gloss", "")) for row in csv.DictReader(f, delimiter="\t")
                if row["id"].startswith(prefix)]


def level_warning(wave):
    peak = float(abs(wave).max()) if len(wave) else 0.0
    if peak >= 0.99:
        return "too loud: it clipped. Move back from the microphone or turn the input level down, then redo it."
    if peak < 0.1:
        return "very quiet. Move closer to the microphone or turn the input level up, then redo it."
    return None


def record_take(sd, samplerate, device):
    frames = queue.Queue()
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", device=device,
                        callback=lambda data, *_: frames.put(data.copy())):
        input("  recording... press Enter to stop ")
    import numpy as np
    chunks = []
    while not frames.empty():
        chunks.append(frames.get())
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype="float32")


def main():
    parser = argparse.ArgumentParser(description="Record the lines of the recording script, one file per line.")
    parser.add_argument("--script", default=os.path.join("recording", "script.tsv"))
    parser.add_argument("--session", type=int, required=True, help="session number (100 lines each)")
    parser.add_argument("-o", "--output", required=True, help="folder for the WAV files")
    parser.add_argument("--samplerate", type=int, default=48000)
    parser.add_argument("--device", default=None, help="input device name or number (see: python -m sounddevice)")
    args = parser.parse_args()

    import sounddevice as sd
    import soundfile

    os.makedirs(args.output, exist_ok=True)
    lines = session_lines(args.script, args.session)
    todo = [line for line in lines if not os.path.exists(os.path.join(args.output, f"{line[0]}.wav"))]
    print(f"Session {args.session}: {len(lines)} lines, {len(lines) - len(todo)} already recorded.\n")
    for number, (line_id, text, gloss) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {line_id}\n\n    {text}\n" + (f"    ({gloss})\n" if gloss else ""))
        action = input("  Enter = record, s = skip, q = quit: ").strip().lower()
        if action == "q":
            break
        if action == "s":
            continue
        while True:
            wave = record_take(sd, args.samplerate, args.device)
            print(f"  {len(wave) / args.samplerate:.1f} s")
            warning = level_warning(wave)
            if warning:
                print("  " + warning)
            action = input("  Enter = keep, p = play, r = redo, q = quit: ").strip().lower()
            while action == "p":
                sd.play(wave, args.samplerate)
                sd.wait()
                action = input("  Enter = keep, p = play, r = redo, q = quit: ").strip().lower()
            if action == "r":
                continue
            if action != "q":
                soundfile.write(os.path.join(args.output, f"{line_id}.wav"), wave, args.samplerate, subtype="PCM_24")
            break
        if action == "q":
            break
        print()
    done = sum(os.path.exists(os.path.join(args.output, f"{line_id}.wav")) for line_id, _, _ in lines)
    print(f"\nSession {args.session}: {done} of {len(lines)} lines recorded in {args.output}")


if __name__ == "__main__":
    main()
