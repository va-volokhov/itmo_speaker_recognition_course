import scipy.io.wavfile as spiowav
from audiofeatures import Audiofeatures
from dataprep import download_dataset, extract_dataset, download_protocol, load_model


# Download VoxCeleb1 (test set)
with open('./lists/datasets.txt', 'r') as f:
    lines = f.readlines()

# download_dataset(lines, user='voxceleb1902', password='nx0bl2v2', save_path='./')

# Extract VoxCeleb1 test set
# extract_dataset(save_path='./voxceleb1_test', fname='vox1_test_wav.zip')

wav_file = "./voxceleb1_test/wav/id10270/5r0dWxy17C8/00001.wav"

rate, sig = spiowav.read(wav_file)

af = Audiofeatures('MFCC')
sig = af.preemphasis(sig)
