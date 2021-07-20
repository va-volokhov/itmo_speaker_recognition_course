# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script downloads the VoxCeleb1 test dataset and model's weights
# Requirement: wget running on a Linux system 


# Import of modules
import os
import subprocess
import hashlib
import tarfile
from zipfile import ZipFile
from collections import OrderedDict

import torch


def md5(fname):
    """
    Estimate md5 sum
    """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def download_dataset(lines, user, password, save_path, reload=False):
    """
    Download datasets from lines with wget
    :param lines: list of datasest to load in format <http_link> <md5 sum>\n
    :param user: user_name
    :param password:
    :param save_path: path to folder
    :param reload: rewrite if file exists
    """

    for line in lines:
        url = line.strip().split(' ')[0]
        md5gt = line.strip().split(' ')[1]
        outfile = url.split('/')[-1]

        out = 0
        # Download files if needed
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s --user %s --password %s -O %s/%s'%(url, user, password, save_path, outfile), shell=True)

        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        # Check MD5
        md5ck = md5('%s/%s'%(save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        
        else:
            raise Warning('Checksum failed %s.'%outfile)


def download_protocol(lines, save_path, reload=False):
    # Download with wget

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        out = 0
        # Download files
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)
        
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        print('File %s is downloaded.'%outfile)


def extract_dataset(save_path, fname):
    # Extract zip files
    
    if fname.endswith(".tar.gz"):
        
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(save_path)
    
    elif fname.endswith(".zip"):
        
        with ZipFile(fname, 'r') as zf:
            zf.extractall(save_path)

    print('Extracting of %s is successful.'%fname)


def load_model(model, lines, save_path, reload=False):
    # Load model's weights
    
    if not os.path.exists(save_path):
        os.mkdir(save_path, mode=0o777)

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        out = 0

        # Download files
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)
        
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        print('File %s is downloaded.'%outfile)
        
    checkpoint = torch.load(os.path.join(save_path, 'baseline_v2_ap.model'))
    
    model_weight = OrderedDict()

    for key in checkpoint.keys():
        
        if '__S__' in key:
            model_weight[key[6:]] = checkpoint[key]
            
    model.load_state_dict(model_weight)
    
    return model