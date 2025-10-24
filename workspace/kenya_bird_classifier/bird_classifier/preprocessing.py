import librosa
import numpy as np


def preprocess(signal, sample_r):
  signal = librosa.effects.preemphasis(signal)
  signal = librosa.util.normalize(signal)
  signal, _ = librosa.effects.trim(signal, top_db=20)
  mel = librosa.feature.melspectrogram(y=signal, sr=sample_r, n_fft=2048, hop_length=512,
                                        n_mels=128, fmin=280, fmax=7600)
  mel_db = librosa.power_to_db(mel, ref=np.max)
  return mel_db, signal