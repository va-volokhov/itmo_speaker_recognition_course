# Exercises in order to perform laboratory work


# Import of modules
from typing import Tuple, List, Any, Tuple, Dict, Optional, AnyStr, Path
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm: Path, signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Function to read rttm files and generate VAD's markup in samples

    :param path_to_rttm: path to rttm markup file
    :param signal: raw input signal
    :param fs: sampling frequency
    :return: vad_markup: VAD's markup in samples

    TODO: Here you should read rttm files and generate VAD's markup in samples
    """

    vad_markup = np.zeros(len(signal)).astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################

    return vad_markup


def framing(signal: np.ndarray, window: int = 320, shift: int = 160) -> np.ndarray:
    """
    Function creates frames from signal

    :param signal: raw input signal
    :param window: size of sliding window in samples
    :param shift: size of the sliding step in samples
    :return: frames [n_frames x win_size]

    TODO: Here you need to prepare framed signal
    """
    shape = (int((signal.shape[0] - window) / shift + 1), window)
    frames = np.zeros().astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################
    return frames


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """
    Function to compute frame energies

    :param frames: matrix of frames of the signal [n_frames x win_size]
    :return: E: energies vector [n_frames x 1]

    TODO: Here you need to compute time energies from framed signal
    """
    e = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################

    return e


def norm_energy(e: np.ndarray) -> np.ndarray:
    """
    Function normalizes energy by mean energy and energy standard deviation

    :param e: signal sliding frames energies vector [n_frames x 1]
    :return: e_norm: normalised energies vector [n_frames x 1]

    TODO: Here you need to normalize time energies
    """
    e_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################

    return e_norm


def gmm_train(e: np.ndarray, gauss_pdf: Any, n_realignment: int) \
        -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Function to train parameters of gaussian mixture model using EM-algorithm

    :param e: time series of input data [n_frames x 1]
    :param gauss_pdf: gaussian probability density function
    :param n_realignment: the number of EM iterations
    :return: w - weights parameters of gmm,
             m - means parameters of gmm,
             sigma - sigmas parameters of gmm

    TODO: Here you need to train GMM model using EM iterations
    """

    # Initialization gaussian mixture models
    w = np.array([0.33, 0.33, 0.33])
    m = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([1.00, 1.00, 1.00])

    g = np.zeros([len(e), len(w)])
    for n in range(n_realignment):
        print('{} iteration'%(n))
        # E-step
        ###########################################################
        # Here is your code

        ###########################################################

        # M-step
        ###########################################################
        # Here is your code

        ###########################################################

    return w, m, sigma


def eval_frame_post_prob(e: np.ndarray,
                         gauss_pdf: Any,
                         w: Union[np.ndarray, float],
                         m: Union[np.ndarray, float],
                         sigma: Union[np.ndarray, float]) -> np.ndarray:
    """
    Function to estimate a posterior probability that frame isn't speech

    :param e: time series of input data [n_frames x 1]
    :param gauss_pdf: gaussian probability density function
    :param w: weights parameters of gmm
    :param m: means parameters of gmm
    :param sigma: sigmas parameters of gmm
    :return: posterior probability time vector gamma_p

    TODO: Here you need to compute a posterior probability vector that each frame isn't speech
    """
    gamma_p = np.zeros(len(e))

    ###########################################################
    # Here is your code

    ###########################################################

    return gamma_p


def energy_gmm_vad(signal: np.ndarray,
                   window: int,
                   shift: int,
                   gauss_pdf: Any,
                   n_realignment: int,
                   vad_thr: float,
                   mask_size_morph_filt: int) -> np.ndarray:
    """
    Function to compute markup energy voice activity detector based of gaussian mixtures model

    :param signal: raw input signal
    :param window: size of sliding window in samples
    :param shift: size of the sliding step in samples
    :param gauss_pdf: gaussian probability density function
    :param n_realignment: number of EM iterations
    :param vad_thr: posterior threshold
    :param mask_size_morph_filt: the size of mask for Morphology Filter
    :return: vad_markup_real: VAD's markup in samples
    """

    # Squared signal
    squared_signal = signal ** 2

    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)

    # Sum frames to get energy
    e = frame_energy(frames)

    # Normalize the energy
    e_norm = norm_energy(e)

    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(e_norm, gauss_pdf, n_realignment=n_realignment)

    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(e_norm, gauss_pdf, w, m, sigma)

    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')
    # Compute sample VAD's markup
    vad_markup_real = np.zeros(len(signal)).astype('float32')
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx * shift:shift + idx * shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real) * shift - len(signal):] = vad_frame_markup_real[-1]

    # Morphology Filters
    # close filter
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt))
    # open filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt))

    return vad_markup_real


def reverb(signal: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    """
    Function to create reverberation effect

    :param signal: raw input signal
    :param impulse_response: impulse response for augmentation
    :return: signal_reverb: reverberated signal

    TODO: Here you need to reverberate input raw signal with target impulse response

    """

    signal_reverb = np.zeros(len(signal)).astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################

    return signal_reverb


def awgn(signal: np.ndarray, sigma_noise: float) -> np.ndarray:
    """
    Function to add white gaussian noise to signal

    :param signal: raw input signal
    :param sigma_noise: sigma of noise to add
    :return: signal_noise: noised signal

    TODO: Here you need to make additive white gaussian noise augmentation
    """
    signal_noise = np.zeros(len(signal)).astype('float32')

    ###########################################################
    # Here is your code

    ###########################################################

    return signal_noise


