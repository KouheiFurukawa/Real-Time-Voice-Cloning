import os
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm

from musicnet_utils import musicnet_label

source_dir = '/groups/1/gcc50521/furukawa/musicnet/*'
sec = 10
target_dir = '/groups/1/gcc50521/furukawa/musicnet_wav_10sec'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for file in tqdm(glob(source_dir + '/*.wav')):
    label = musicnet_label(file)
    filename = file.split('/')[-1][:-4]
    try:
        wav, sr = librosa.load(file, 16000)
        if np.any(np.isnan(wav)):
            continue
        for i in range(len(wav) // (sr * sec)):
            x = wav[i * (sr * sec):(i + 1) * (sr * sec)]
            if len(x) < 81920:
                continue
            # np.save(target_dir + '/{}_{}.npy'.format(filename, str(i)), x)
            os.makedirs(target_dir + '/' + str(label), exist_ok=True)
            sf.write(target_dir + '/' + str(label) + '/{}_{}.wav'.format(filename, str(i)), x, 16000)
    except:
        print(file)

print(len(glob(target_dir + '/*.wav')))
