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
    """
    
    vad_markup = np.zeros(len(signal)).astype('float32')
    
    with open(path_to_rttm, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip().split(' ')
        
        start = float(line[3])*fs
        dur = float(line[4])*fs
        end = start + dur
        
        vad_markup[int(start):int(end)] = 1
    
    return vad_markup


def framing(signal: np.ndarray, window: int=320, shift: int=160) -> np.ndarray:
    """
    Function creates frames from signal

    :param signal: raw input signal
    :param window: size of sliding window in samples
    :param shift: size of the sliding step in samples
    :return: frames [n_frames x win_size]
    """
    shape = (int((signal.shape[0] - window)/shift + 1), window)
    strides = (signal.strides[0]*shift, signal.strides[0])
    
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    return frames


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """
    Function to compute frame energies

    :param frames: matrix of frames of the signal [n_frames x win_size]
    :return: E: energies vector [n_frames x 1]
    """
    e = frames.sum(axis=1)
    
    return e


def norm_energy(e: np.ndarray) -> np.ndarray:
    """
    Function normalizes energy by mean energy and energy standard deviation

    :param e: signal sliding frames energies vector [n_frames x 1]
    :return: e_norm: normalised energies vector [n_frames x 1]
    """
    e_norm = np.copy(e)
    e_norm -= e_norm.mean()
    e_norm /= (e_norm.std() + 1e-10)
    
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
    """

    # Initialization gaussian mixture models
    w = np.array([0.33, 0.33, 0.33])
    m = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([1.00, 1.00, 1.00])

    g = np.zeros([len(e), len(w)])
    for n in range(n_realignment):

        # E-step
        for j in range(len(m)):
            g[:, j] = gauss_pdf(e, m[j], sigma[j])

        g_norm = np.zeros(g.shape[0])
        for j in range(len(m)):
            g_norm = g_norm + w[j]*g[:, j]

        for j in range(len(m)):
            g[:, j] = w[j]*g[:, j]/(g_norm + 1e-10)

        # M-step
        w = g.sum(axis=0)/len(e)

        m = np.zeros([len(e), len(w)])
        for j in range(len(m)):
            m[:, j] = g[:, j] * e

        m = m.sum(axis=0) / len(e) / (w + 1e-10)

        s = np.zeros([len(e), len(w)])
        for j in range(len(m)):
            s[:, j] = g[:, j] * (e - m[j]) ** 2

        sigma = s.sum(axis=0) / len(e) / (w + 1e-10)
        sigma = np.sqrt(sigma)
        
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
    """
    g = np.zeros([len(e), len(w)])
    
    for j in range(len(m)):
        g[:, j] = gauss_pdf(e, m[j], sigma[j])

    g_norm = np.zeros(g.shape[0])
    for j in range(len(m)):
        g_norm = g_norm + w[j]*g[:, j]

    gamma_p = w[0]*g[:, 0]/(g_norm + 1e-10)

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
    squared_signal = signal**2
    
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
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
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
    """
    
    signal_reverb = scipy.signal.convolve(signal, impulse_response, mode='full')
    signal_reverb = signal_reverb/np.abs(signal_reverb).max()
    signal_reverb = signal_reverb[:-(len(impulse_response) - 1)]
    
    return signal_reverb


def awgn(signal: np.ndarray, sigma_noise: float) -> np.ndarray:
    """
    Function to add white gaussian noise to signal

    :param signal: raw input signal
    :param sigma_noise: sigma of noise to add
    :return: signal_noise: noised signal
    """
    signal_noise = signal + sigma_noise*np.random.randn(len(signal)).astype('float32')
    signal_noise = signal_noise/np.abs(signal_noise).max()
    
    return signal_noise
