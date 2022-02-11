# Exercises for laboratory work


# Import of modules
import numpy as np
from scipy.fftpack import dct
from typing import Tuple, List, Any, Tuple, Dict, Optional, AnyStr


def split_meta_line(line: str, delimiter: str = ' ') -> Tuple[AnyStr, AnyStr, AnyStr]:
    """
    Function parses the line from meta file with "Speaker_ID Gender Path"

    :param line: lines of metadata
    :param delimiter: delimeter
    :return: speaker_id: speaker IDs: gender: gender: file_path: path to file

    TODO: First you need to prepare split_meta_line function for meta line parsing
          line format is "Speaker_ID Gender Path"
    """

    ###########################################################
    # Here is your code

    ###########################################################

    return speaker_id, gender, file_path


def preemphasis(signal: np.ndarray, pre_emphasis: float = 0.97) -> np.ndarray:
    """
    Function preemphases input signal with pre_emphasis coeffitient

    :param signal: input signal
    :param pre_emphasis: preemphasis coeffitient
    :return: emphasized_signal: signal after pre-emphasis procedure

    TODO: Here you need to preemphasis input signal with pre_emphasis coeffitient
    """
    ###########################################################
    # Here is your code

    ###########################################################

    return emphasized_signal


def framing(emphasized_signal: np.ndarray, sample_rate: int=16000, frame_size: float=0.025, frame_stride: float=0.01) \
        -> np.ndarray:
    """
    Function performs framing of the input signal emphasized_signal with sample rate sample_rate with hamming windowing

    :param emphasized_signal: signal after pre-emphasis procedure
    :param sample_rate: signal sampling rate
    :param frame_size: sliding window size in seconds
    :param frame_stride: step
    :return: frames: output matrix [nframes x sample_rate*frame_size]

    TODO: Here you need to perform framing of the input signal emphasized_signal with sample rate sample_rate
          Please use hamming windowing
    """

    # convertion from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    # pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    ###########################################################
    # Here is your code to compute frames

    ###########################################################

    return frames


def power_spectrum(frames: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    Function computes power spectrum of the framed signal

    :param frames: framed signal [n_frames x win_size]
    :param n_fft: number of fft bins
    :return: pow_frames: framed signal power spectrum
    """

    # magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    # power Spectrum

    ###########################################################
    # Here is your code to compute pow_frames

    ###########################################################
    return pow_frames


def compute_fbank_filters(n_filt=40, sample_rate=16000, n_fft=512):
    """
    Function computes Mel Filter Bank features
    :param n_filt: number of filters
    :param sample_rate: signal sampling rate
    :param n_fft: number of fft bins in power spectrum
    :return: fbank [nfilt x (NFFT/2+1)]

    TODO: Here you need to compute fbank filters (FBs) for special case (sample_rate & n_fft)
    """

    low_freq_mel = 0
    high_freq = sample_rate / 2

    ###########################################################
    # Here is your code to convert Convert Hz to Mel:
    # high_freq -> high_freq_mel

    ###########################################################
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)

    ###########################################################
    # Here is your code to convert Convert Mel to Hz:
    # mel_points -> hz_points

    ###########################################################

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate)
    fbank = np.zeros((n_filt, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filt + 1):
        # left
        f_m_minus = int(bin_points[m - 1])
        # center
        f_m = int(bin_points[m])
        # right
        f_m_plus = int(bin_points[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    return fbank


def compute_fbanks_features(pow_frames: np.ndarray, fbank: np.ndarray) -> np.ndarray:
    """
    Function computes fbank features using power spectrum frames and suitable fbanks filters

    :param pow_frames: framed signal power spectrum, matrix [nframes x sample_rate*frame_size]
    :param fbank: matrix of the fbank filters [nfilt x (NFFT/2+1)] where NFFT: number of fft bins in power spectrum
    :return: filter_banks_features: log mel FB energies matrix [nframes x nfilt]

    TODO: You need to compute fbank features using power spectrum frames and suitable fbanks filters
    """

    ###########################################################
    # Here is your code to compute filter_banks_features

    ###########################################################
    # for numerical stability
    filter_banks_features = np.where(filter_banks_features == 0, np.finfo(float).eps, filter_banks_features)
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features


def compute_mfcc(filter_banks_features: np.ndarray, num_ceps: int=20) -> np.ndarray:
    """
    Function computes MFCCs features using precomputed log mel FB energies matrix

    :param filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    :param num_ceps: number of cepstral components for MFCCs
    :return: mfcc: mel-frequency cepstral coefficients (MFCCs)

    TODO: Here you need to compute MFCCs features using precomputed log mel FB energies matrix
    """

    ###########################################################
    # Here is your code to compute mfcc features

    ###########################################################

    return mfcc


def mvn_floating(features: np.ndarray, lc: int, rc: int, unbiased: bool = False) -> np.ndarray:
    """
    Function does mean variance normalization of the input features

    :param features: features matrix [nframes x nfeats]
    :param lc: the number of frames to the left defining the floating
    :param rc: the number of frames to the right defining the floating
    :param unbiased: biased or unbiased estimation of normalising sigma
    :return: normalised_features: normalised features matrix [nframes x nfeats]

    TODO: Here you need to do mean variance normalization of the input features
    """

    nframes, dim = features.shape
    lc = min(lc, nframes - 1)
    rc = min(rc, nframes - 1)
    n = (np.r_[np.arange(rc + 1, nframes), np.ones(rc + 1) * nframes]
         - np.r_[np.zeros(lc), np.arange(nframes - lc)])[:, np.newaxis]
    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)
    f = (np.r_[f[rc:], np.repeat(f[[-1]], rc, axis=0)] - np.r_[np.zeros((lc + 1, dim)), f[:-lc - 1]]) / n
    s = (np.r_[s[rc:], np.repeat(s[[-1]], rc, axis=0)] - np.r_[np.zeros((lc + 1, dim)), s[:-lc - 1]]
         ) / (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)

    ###########################################################
    # Here is your code to compute normalised features

    ###########################################################

    normalised_features[s == 0] = 0

    return normalised_features
