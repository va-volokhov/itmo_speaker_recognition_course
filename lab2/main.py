import pathlib
import sys

project_root = str(pathlib.Path(__file__).absolute().parents[1])
sys.path.append(project_root)

from common import download_dataset, extract_dataset, download_protocol, load_model


if __name__ == '__main__':

    # Download VoxCeleb1 (test set)
    with open('./data/lists/datasets.txt', 'r') as f:
        lines = f.readlines()

    download_dataset(lines, user='voxceleb1902', password='nx0bl2v2', save_path='../data', reload=True)
    # Extract VoxCeleb1 test set
    extract_dataset(save_path='./data/voxceleb1_test', fname='vox1_test_wav.zip')

    # Download VoxCeleb1-O cleaned protocol
    with open('./data/lists/protocols.txt', 'r') as f:
        lines = f.readlines()

    download_protocol(lines, save_path='./data/voxceleb1_test')