import librosa


def load_sample(path, center_time, dur=3.0, sr_target=22050):
  return librosa.load(path, sr=sr_target, offset=max(0, center_time - dur/2), duration=dur)