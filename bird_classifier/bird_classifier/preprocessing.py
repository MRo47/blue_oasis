import librosa
import noisereduce as nr
import numpy as np
from numpy.typing import NDArray


def preprocess(y: NDArray, sr: int, 
               noise_prop_decrease: float = 0.9,
               noise_stationary: bool = True,
               pre_emphasis: float = 0.97,
               normalize: bool = True):
	"""
	Preprocesses a raw audio signal.

	First, noise reduction is applied using the `noisereduce` library.
	Then, pre-emphasis is applied to the signal using `librosa`.
	If `normalize` is True, the signal is normalized using `librosa`.

	Parameters
	----------
	y : NDArray
			The raw audio signal.
	sr : int
			The sampling rate of the signal.
	noise_prop_decrease : float, optional
			The proportion of the signal that is treated as noise. Default is 0.9.
	noise_stationary : bool, optional
			Whether to assume stationary noise. Default is True.
	pre_emphasis : float, optional
			The coefficient for pre-emphasis. Default is 0.97.
	normalize : bool, optional
			Whether to normalize the signal. Default is True.

	Returns
	-------
	y : NDArray
			The preprocessed audio signal.
	"""
	y = nr.reduce_noise(y=y, sr=sr, prop_decrease=noise_prop_decrease, 
											stationary=noise_stationary)
	y = librosa.effects.preemphasis(y, coef=pre_emphasis)
	if normalize:
		y = librosa.util.normalize(y)
	return y

def extract_mel_db_spec(y: NDArray, sr: int,
												n_fft: int = 2048, hop_length: int = 512,
												n_mels: int = 128, fmin: int = 280,
												fmax: int = 7600):
	"""
	Extracts the mel spectrogram and scales it to decibels.

	Parameters
	----------
	y : NDArray
		The preprocessed audio signal.
	sr : int
		The sampling rate of the signal.
	n_fft : int, optional
		The number of frequency bins in the mel spectrogram. Default is 2048.
	hop_length : int, optional
		The number of samples between adjacent frames in the mel spectrogram. Default is 512.
	n_mels : int, optional
		The number of mel bins in the mel spectrogram. Default is 128.
	fmin : int, optional
		The minimum frequency of the mel spectrogram. Default is 280 Hz.
	fmax : int, optional
		The maximum frequency of the mel spectrogram. Default is 7600 Hz.

	Returns
	-------
	mel_db : NDArray
		The mel spectrogram in decibels.
	y : NDArray
		The preprocessed audio signal.
	"""
	mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
																			hop_length=hop_length, n_mels=n_mels, 
																			fmin=fmin, fmax=fmax)
	mel_db = librosa.power_to_db(mel, ref=np.max)
	return mel_db, y