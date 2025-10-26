import audiomentations as am
import numpy as np

"""composed time augmentations for bird classification

1. adds backgorund noise resembling wind/rain and low rumble at 0.5 probability
2. shifts the audio signal left or right fractionally +- 0.3 at 0.5 probability
3. pitch shift +- 4 semitones at 0.3 probability
4. speed shift +- 20% at 0.3 probability
5. volume variation +- 12dB at 0.5 probability
"""
time_augmentations = am.Compose([
    # Add background noise with a 50% probability
    am.OneOf([
        # add noise from dataset
        # am.AddBackgroundNoise(sounds_path=kenya_noises_path, min_snr_in_db=5.0, max_snr_in_db=35.0),
        # add synthetic pink noise (simulates wind/environmental hum)
        am.AddColorNoise(min_snr_db=5.0, max_snr_db=35.0, min_f_decay=-3.01, max_f_decay=-3.01),
        # add synthetic brown noise (simulates heavy rain/rumble)
        am.AddColorNoise(min_snr_db=5.0, max_snr_db=35.0, min_f_decay=-6.02, max_f_decay=-6.02),
    ], p=0.5),

    # like numpy roll
    am.Shift(min_shift=-0.3, max_shift=0.3, shift_unit='fraction', p=0.5),

    # pitch variation
    am.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
    
    # speed variation
    am.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),

    # volume variation
    am.Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
])



def time_domain_augment(y, sr=22050):
	"""
	Applies time domain augmentations to an audio signal.
	1. adds backgorund noise resembling wind/rain and low rumble at 0.5 probability
	2. shifts the audio signal left or right fractionally +- 0.3 at 0.5 probability
	3. pitch shift +- 4 semitones at 0.3 probability
	4. speed shift +- 20% at 0.3 probability
	5. volume variation +- 12dB at 0.5 probability

	Parameters
	----------
	y : NDArray
			The audio signal to be augmented.
	sr : int, optional
			The sampling rate of the signal. Default is 22050.

	Returns
	-------
	The augmented audio signal.
	"""
	return time_augmentations(samples=y, sample_rate=sr)

def spec_augment(mel_spec_db, freq_mask_width=10, time_mask_width=10):
	"""
	Applies frequency and time masking to a mel spectrogram.

	Parameters
	----------
	mel_spec_db : NDArray
		The mel spectrogram to be augmented.
	freq_mask_width : int, optional
		The width of the frequency mask. Default is 10.
	time_mask_width : int, optional
		The width of the time mask. Default is 10.

	Returns
	-------
	mel_spec_db : NDArray
		The augmented mel spectrogram.
	"""
	mel_spec_db = mel_spec_db.copy()

	min_db = mel_spec_db.min()

	# Frequency masking
	num_freq_bins = mel_spec_db.shape[0]
	f = np.random.randint(0, freq_mask_width)
	f0 = np.random.randint(0, num_freq_bins - f)
	mel_spec_db[f0:f0+f, :] = min_db

	# Time masking
	num_time_steps = mel_spec_db.shape[1]
	t = np.random.randint(0, time_mask_width)
	t0 = np.random.randint(0, num_time_steps - t)
	mel_spec_db[:, t0:t0+t] = min_db

	return mel_spec_db