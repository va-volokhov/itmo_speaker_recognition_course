# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
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

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)

    all_scores           = np.array(all_scores)
    all_scores_sort_idxs = np.argsort(all_scores)

    all_scores_sort   = []
    ground_truth_sort = []
    for idx in all_scores_sort_idxs:
        all_scores_sort.append(all_scores[idx])
        ground_truth_sort.append(all_labels[idx])

    all_scores_sort   = np.array(all_scores_sort)
    ground_truth_sort = np.array(ground_truth_sort, dtype='bool')

    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR           = np.log(tar_gauss_pdf/imp_gauss_pdf)
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                     # decision
        
        err = (solution != ground_truth_sort)                         # error vector
        
        fnr_thr[idx] = np.sum(err[ground_truth_sort])/len(tar_scores) # prob. of Type I error P(Dimp|Htar), false negative rate (FNR)
    
    thr_idx  = np.argmin(np.abs(fnr_thr - fnr))                # search of threshold for a given FNR
    solution = LLR > LLR[thr_idx]                              # decision
    err      = (solution != ground_truth_sort)                 # error vector
    fpr      = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
    
    return LLR[thr_idx], fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    len_thr = len(LLR)
    tpr_thr = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)
    AC      = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        ts  = (solution == ground_truth_sort)                          # true solution vector
        err = (solution != ground_truth_sort)                          # error vector
        
        tpr_thr[idx] = np.sum(ts [ ground_truth_sort])/len(tar_scores) # true positive ratio (TPR)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        tnr_thr[idx] = np.sum(ts[ ~ground_truth_sort])/len(imp_scores) # true negative ratio (TNR)
        
        AC[idx]  = C00*tpr_thr[idx]*P_Htar       + \
                   C10*fnr_thr[idx]*P_Htar       + \
                   C01*fpr_thr[idx]*(1 - P_Htar) + \
                   C11*tnr_thr[idx]*(1 - P_Htar)                       # Bayes' risk (average cost)

    # Plot average cost
    plot(LLR, AC, color='blue')
    xlabel('$LLR$'); ylabel('$\overline{C}$'); title('Average cost'); grid(); show()    
    
    AC_idx = np.argmin(AC) # argmin of Bayes' risk
    
    return LLR[AC_idx], fnr_thr[AC_idx], fpr_thr[AC_idx], AC[AC_idx]

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    len_thr        = len(LLR)
    len_P_Htar_thr = len(P_Htar_thr)
    tpr_thr        = np.zeros(len_thr)
    fnr_thr        = np.zeros(len_thr)
    fpr_thr        = np.zeros(len_thr)
    tnr_thr        = np.zeros(len_thr)
    AC             = np.zeros([len_thr, len_P_Htar_thr])

    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        ts  = (solution == ground_truth_sort)                          # true solution vector
        err = (solution != ground_truth_sort)                          # error vector
        
        tpr_thr[idx] = np.sum(ts [ ground_truth_sort])/len(tar_scores) # true positive ratio (TPR)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        tnr_thr[idx] = np.sum(ts[ ~ground_truth_sort])/len(imp_scores) # true negative ratio (TNR)
        
        for idy in range(len_P_Htar_thr):
            AC[idx, idy] = C00*tpr_thr[idx]*P_Htar_thr[idy]       + \
                           C10*fnr_thr[idx]*P_Htar_thr[idy]       + \
                           C01*fpr_thr[idx]*(1 - P_Htar_thr[idy]) + \
                           C11*tnr_thr[idx]*(1 - P_Htar_thr[idy])      # Bayes' risk (average cost)
    
    # Plot average cost
    imshow(AC[18705:18905, :], extent=[P_Htar_thr[0], P_Htar_thr[999], LLR[18905], LLR[18705]])
    xlabel('$P(H_0)$'); ylabel('$LLR$'); title('Average cost surface (top view)'); show()
    
    AC_P_Htar_max = np.zeros(len_thr)
    for idx in range(len_thr):
        AC_P_Htar_max[idx] = np.amax(AC[idx, :])
    
    AC_min_max_idx = np.argmin(AC_P_Htar_max)
    
    AC_thr_min = np.zeros(len_P_Htar_thr)
    for idx in range(len_P_Htar_thr):
        AC_thr_min[idx] = np.amin(AC[:, idx])
    
    AC_max_min_idx = np.argmax(AC_thr_min)
    
    solution = LLR > LLR[AC_min_max_idx]                       # decision
    err      = (solution != ground_truth_sort)                 # error vector
    fnr      = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)       
    fpr      = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
    
    return LLR[AC_min_max_idx], fnr, fpr, AC[AC_min_max_idx, AC_max_min_idx], P_Htar_thr[AC_max_min_idx]