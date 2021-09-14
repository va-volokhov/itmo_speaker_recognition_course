# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script allows to download data from dataset


# Import of modules
import os

import numpy
import soundfile
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Load wav file
    
    max_audio = max_frames*160 + 240

    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage  = max_audio - audiosize + 1 
        audio     = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize - max_audio))])
    
    feats = []
    
    if evalmode and max_frames == 0:
        feats.append(audio)
    
    else:
        
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat

class test_dataset_loader(Dataset):
    # Test dataset loader
    
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        
        audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        
        return len(self.test_list)