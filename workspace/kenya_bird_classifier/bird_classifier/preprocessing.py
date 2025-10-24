import librosa
import noisereduce as nr
import numpy as np


def preprocess(signal, sample_r):
  signal = nr.reduce_noise(y=signal, sr=sample_r, prop_decrease=0.9, stationary=True)
  signal, _ = librosa.effects.trim(signal, top_db=30)
  signal = librosa.effects.preemphasis(signal)
  signal = librosa.util.normalize(signal)
  mel = librosa.feature.melspectrogram(y=signal, sr=sample_r, n_fft=2048, hop_length=512,
                                        n_mels=128, fmin=280, fmax=7600)
  mel_db = librosa.power_to_db(mel, ref=np.max)
  return mel_db, signal