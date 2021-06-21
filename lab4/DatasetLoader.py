# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script allows to download data from dataset, to extract embeddings, to compute scores between enroll and test speaker models for performing of test procedure


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

def extract_features(model, test_loader):
    # Extract features for every waveform

    feats = {}

    for idx, data in enumerate(test_loader):
        inp1 = data[0][0].cuda()
        
        with torch.no_grad():
            ref_feat = model(inp1).detach().cpu()
        
        feats[data[1][0]] = ref_feat

    return feats

def compute_scores(feats, lines):
    # Compute scores

    all_scores = []
    all_labels = []
    all_trials = []

    for idx, line in enumerate(lines):

        data = line.split()

        ref_feat = feats[data[1]].cuda()
        com_feat = feats[data[2]].cuda()

        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

        score = -1*numpy.mean(dist)
        
        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trials.append(data[1]+" "+data[2])

    return all_scores, all_labels, all_trials