import librosa


def load_sample(path, center_time, dur=3.0, sr_target=22050):
  """
  Load an audio sample from disk.

  Parameters
  ----------
  path : str
      Path to the audio file.
  center_time : float
      Time in seconds for the center of the sample.
  dur : float, optional
      Duration of the sample in seconds. Defaults to 3.0.
  sr_target : int, optional
      Target sample rate. Defaults to 22050.

  Returns
  -------
  sample : tuple
      A tuple containing the loaded audio sample and its sample rate.
  """
  return librosa.load(path, sr=sr_target, offset=max(0, center_time - dur/2), duration=dur)