import csv
import pathlib
import random

import click
import librosa
import numpy as np
import soundfile
import tqdm
from textgrid import TextGrid

def build_dataset(wavs, tg, dataset, skip_silence_insertion, wav_subtype):
    wavs = pathlib.Path(wavs)
    tg_dir = pathlib.Path(tg)
    del tg
    dataset = pathlib.Path(dataset)
    filelist = list(wavs.glob('*.wav'))

    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / 'wavs').mkdir(exist_ok=True)
    transcriptions = []
    samplerate = 44100
    min_sil = int(0.1 * samplerate)
    max_sil = int(0.5 * samplerate)
    for wavfile in tqdm.tqdm(filelist):
        y, _ = librosa.load(wavfile, sr=samplerate, mono=True)
        tgfile = tg_dir / wavfile.with_suffix('.TextGrid').name
        tg = TextGrid()
        tg.read(str(tgfile))
        ph_seq = [ph.mark for ph in tg[1]]
        ph_dur = [ph.maxTime - ph.minTime for ph in tg[1]]
        ph_seq = ' '.join(ph_seq)
        ph_dur = ' '.join([str(round(d, 6)) for d in ph_dur])
        soundfile.write(dataset / 'wavs' / wavfile.name, y, samplerate, subtype=wav_subtype)
        transcriptions.append({'name': wavfile.stem, 'ph_seq': ph_seq, 'ph_dur': ph_dur})

    with open(dataset / 'transcriptions.csv', 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur'])
        writer.writeheader()
        writer.writerows(transcriptions)

    print(f'All wavs and transcriptions saved in {dataset}')


if __name__ == '__main__':
    build_dataset()
