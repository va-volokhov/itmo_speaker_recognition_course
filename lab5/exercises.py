# Exercises in order to perform laboratory work

# Import of modules
import numpy as np
from math import sqrt
import itertools
import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from sklearn.metrics.pairwise import cosine_similarity
from cllr_loss import CllrLoss
from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    """
    Get target and impostors scores based on the labels
    """

    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):

        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])

        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores


def plot_histograms_2sets(all_scores_1, all_labels_1,
                          all_scores_2, all_labels_2,
                          names=['in_domain', 'out_of_domain']):
    """

    """
    tar_scores_1, imp_scores_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    tar_scores_2, imp_scores_2 = get_tar_imp_scores(all_scores_2, all_labels_2)

    # Plot histograms for target and impostor scores
    min_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).min()
    max_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).max()

    hist(tar_scores_1, int(sqrt(len(tar_scores_1))), histtype='step', color='green',
         range=(min_scores, max_scores))
    hist(imp_scores_1, int(sqrt(len(imp_scores_1))), histtype='step', color='red',
         range=(min_scores, max_scores))
    hist(tar_scores_2, int(sqrt(len(tar_scores_2))), histtype='step', color='blue',
         range=(min_scores, max_scores))
    hist(imp_scores_2, int(sqrt(len(imp_scores_2))), histtype='step', color='cyan',
         range=(min_scores, max_scores))
    xlabel('$s$');
    ylabel('$\widehat{W}(s|H_0)$, $\widehat{W}(s|H_1)$');
    title('VoxCeleb1-O (cleaned), histograms');
    legend(list('{}_{}'.format(a[0], a[1]) for a in itertools.product(names, ['tar', 'imp'])))
    grid()
    show()

    # Compute Equal Error Rate
    EER_1,thresh_EER_1, DCF_1, thresh_DCF_1 = get_eer(tar_scores_1, imp_scores_1)
    EER_2, thresh_EER_2, DCF_2, thresh_DCF_2 = get_eer(tar_scores_2, imp_scores_2)

    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f}, DCF: {3:.3f} ".format(names[0],
                                                                                  EER_1, thresh_EER_1,
                                                                                  DCF_1))
    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f}, DCF: {3:.3f}".format(names[1],
                                                                                              EER_2, thresh_EER_2,
                                                                                              DCF_2))


def mean_embd_norm(test_embds, adapt_embds):
    """
    Apply mean embedding normalization
    """
    test_embds_adapted = {}
    adapt_embds_list = [adapt_embds[k] for k in adapt_embds.keys()]
    mean_embd = torch.stack(adapt_embds_list).mean(0)
    if len(mean_embd.size()) > 1:
        mean_embd = mean_embd.mean(0)

    for k in test_embds.keys():
        test_embds_adapted[k] = test_embds[k] - mean_embd
    return test_embds_adapted


def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    """
    Performs s-normalization for scores "scores" with the snorm_data
    :param test_data: test embeddings
    :param lines: test protocol
    :param scores: raw scores matrix
    :param adapt_data: data for s-norm (s-norm embeddings)
    :param N_s: top N impostors scrores for s-normalization
    :param eps: epsilon for std
    :return:
            snorm_scores - s-normalized scores
    """

    scores_adapted = []
    all_labels = []
    all_trials = []

    # prepare lists of unique wavs from protocols
    enroll_list = list(set(list([x.strip().split()[1] for x in lines])))
    test_list = list(set(list([x.strip().split()[2] for x in lines])))
    adapt_list = list(adapt_data.keys())

    # prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
    E = []
    for id, enr in enumerate(enroll_list):
        E.append(test_data[enr].squeeze(0).numpy())
    E = np.array(E)

    # prepare tests: save test embds in ndarray [num_wavs x emb_size]
    T = []
    for id, tst in enumerate(test_list):
        T.append(test_data[tst].squeeze(0).numpy())
    T = np.array(T)

    # prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
    A = []
    for id, a in enumerate(adapt_list):
        A.append(adapt_data[a].squeeze(0).numpy())
    A = np.array(A)

    # prepare scores with enroll-vs-adapt set
    scores_e = cosine_similarity(E, A)
    # prepare scores with test-vs-adapt se
    scores_t = cosine_similarity(T, A)
    # sort scores to choose the most hard ones
    scores_e = -np.sort(-scores_e, axis=1)
    scores_t = -np.sort(-scores_t, axis=1)
    # use N_s hard impostors to get mean and std for each enroll and test
    mn_t = scores_t[:, :N_s].mean(1)[:, np.newaxis]
    mn_e = scores_e[:, :N_s].mean(1)[:, np.newaxis]
    std_t = np.std(scores_t[:, :N_s], axis=1)[:, np.newaxis]
    std_e = np.std(scores_e[:, :N_s], axis=1)[:, np.newaxis]

    # create dict for faster operations
    snorm_params_enr = {}
    for id, enr in enumerate(enroll_list):
        snorm_params_enr[enr] = [mn_e[id], std_e[id]]

    snorm_params_tst = {}
    for id, tst in enumerate(test_list):
        snorm_params_tst[tst] = [mn_t[id], std_t[id]]

    for idx, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc='Scoring with s-norm'):
        trial_label, enroll_wav, test_wav = line.split()
        E = test_data[enroll_wav].squeeze(0).numpy()
        T = test_data[test_wav].squeeze(0).numpy()

        E = E.reshape(1, -1)
        T = T.reshape(1, -1)
        score = cosine_similarity(E, T)

        ## start snorm
        mn_e, std_e = snorm_params_enr[enroll_wav]
        mn_t, std_t = snorm_params_tst[test_wav]

        score = (score - mn_e) / (std_e + eps) + (score - mn_t.T) / (std_t.T + eps)
        score = score[0][0]
        ## end snorm

        scores_adapted.append(score)
        all_labels.append(int(trial_label))
        all_trials.append(enroll_wav + " " + test_wav)

    return scores_adapted, all_labels, all_trials


def train_calibration(model, train_set, test_set, system_name, num_epochs=10):
    """
    :return:
    """

    batch_size = 200
    lr = 2

    params = list(model.parameters())

    if len(params) != 0:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
        # optimizer = optim.Adam(params, lr=lr)#, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    else:
        optimizer = None
        scheduler = None

    criterion_verif = CllrLoss()
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=1,
                              shuffle=True)

    for epoch in range(0, num_epochs):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            tar_sc = batch_data['tar_scores']
            imp_sc = batch_data['imp_scores']
            optimizer.zero_grad()

            calib_tar_sc = model(tar_sc.transpose(0, 1)).squeeze()
            calib_imp_sc = model(imp_sc.transpose(0, 1)).squeeze()
            loss = criterion_verif(calib_tar_sc, calib_imp_sc)
            loss.backward()

            optimizer.step()
            # if scheduler is not None:
            #     if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            #         scheduler.step()

            lr_value = optimizer.param_groups[0]['lr']

            print('batch {}, cllr loss {} lr {}'.format(batch_idx + epoch * len(train_loader), loss.item(), lr_value))
    return
