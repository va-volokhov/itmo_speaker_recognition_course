# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
    
    with open(path_to_rttm, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line  = line.strip().split(' ')
        
        start = float(line[3])*fs
        dur   = float(line[4])*fs
        end   = start + dur
        
        vad_markup[int(start):int(end)] = 1
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal
    
    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    strides = (signal.strides[0]*shift, signal.strides[0])
    
    frames  = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    E = frames.sum(axis=1)
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.copy(E)
    E_norm -= E_norm.mean()
    E_norm /= (E_norm.std() + 1e-10)
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):

        # E-step
        for j in range(len(m)):
            g[:, j] = gauss_pdf(E, m[j], sigma[j])

        g_norm = np.zeros(g.shape[0])
        for j in range(len(m)):
            g_norm = g_norm + w[j]*g[:, j]

        for j in range(len(m)):
            g[:, j] = w[j]*g[:, j]/(g_norm + 1e-10)

        # M-step
        w = g.sum(axis=0)/len(E)

        M = np.zeros([len(E), len(w)])
        for j in range(len(m)):
            M[:, j] = g[:, j]*E

        m = M.sum(axis=0)/len(E)/(w + 1e-10)

        S = np.zeros([len(E), len(w)])
        for j in range(len(m)):
            S[:, j] = g[:, j]*(E - m[j])**2

        sigma = S.sum(axis=0)/len(E)/(w + 1e-10)
        sigma = np.sqrt(sigma)
        
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g = np.zeros([len(E), len(w)])
    
    for j in range(len(m)):
        g[:, j] = gauss_pdf(E, m[j], sigma[j])

    g_norm = np.zeros(g.shape[0])
    for j in range(len(m)):
        g_norm = g_norm + w[j]*g[:, j]
            
    return w[0]*g[:, 0]/(g_norm + 1e-10)

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = scipy.signal.convolve(signal, impulse_response, mode='full')
    signal_reverb = signal_reverb/np.abs(signal_reverb).max()
    signal_reverb = signal_reverb[:-(len(impulse_response) - 1)]
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = signal + sigma_noise*np.random.randn(len(signal)).astype('float32')
    signal_noise = signal_noise/np.abs(signal_noise).max()
    
    return signal_noise