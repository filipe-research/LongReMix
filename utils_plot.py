import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins



def plot_guess_view(data,inds_guess_correct, inds_guess_wrong, path, epoch):
    bins = compute_histogram_bins(data, 0.01)

    plt.hist(data,bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='all')
    plt.hist(data[inds_guess_correct], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_correct')
    plt.hist(data[inds_guess_wrong], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_wrong')
    plt.xlabel('Loss');
    plt.ylabel('Number of data')
    plt.legend()
    plt.savefig('%s/guess_loss_histogram_epoch%03d.png' % (path,epoch))
    plt.clf()  

    plt.hist(data,bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='all')
    plt.hist(data[inds_guess_correct],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_correct')
    plt.hist(data[inds_guess_wrong], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_wrong')
    plt.xlabel('prob');
    plt.ylabel('number of data')
    plt.legend()
    plt.savefig('%s/guess_prob_histogram_epoch%03d.png' % (path,epoch))
    plt.clf() 

def plot_guess_view_loss(data,inds_guess_correct, inds_guess_wrong, path, epoch):
    bins = compute_histogram_bins(data, 0.01)

    plt.hist(data,bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='all')
    plt.hist(data[inds_guess_correct], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_correct')
    plt.hist(data[inds_guess_wrong], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_wrong')
    plt.xlabel('Loss');
    plt.ylabel('Number of data')
    plt.legend()
    plt.savefig('%s/guess_loss_histogram_epoch%03d.png' % (path,epoch))
    plt.clf()  

def plot_guess_view_pred(data,inds_guess_correct, inds_guess_wrong, path, epoch):
    bins = compute_histogram_bins(data, 0.01)

    plt.hist(data,bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='all')
    plt.hist(data[inds_guess_correct],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_correct')
    plt.hist(data[inds_guess_wrong], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy_wrong')
    plt.xlabel('prob');
    plt.ylabel('number of data')
    plt.legend()
    plt.savefig('%s/guess_prob_histogram_epoch%03d.png' % (path,epoch))
    plt.clf() 

def plot_histogram_loss_pred(data, inds_clean, inds_noisy, path, epoch):

    bins = compute_histogram_bins(data, 0.01)

    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

    # plt.hist(data, bins=bins, range=(0., 1.), edgecolor='black', color='g')
    # plt.xlabel('Loss');
    # plt.ylabel('Number of data')
    # plt.savefig('%s/histogram_epoch%03d.png' % (path,epoch))
    # plt.clf()

    plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
    if len(inds_noisy) >0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
    plt.xlabel('Loss');
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/sep_loss_epoch%03d.png' % (path,epoch))
    plt.clf()      

    plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
    if len(inds_noisy) >0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
    plt.xlabel('Prob');
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/preds_sep_epoch%03d.png' % (path,epoch))
    plt.clf() 

def plot_histogram_loss(data, inds_clean, inds_noisy, path, epoch):

    bins = compute_histogram_bins(data, 0.01)

    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

    plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
    if len(inds_noisy) >0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
    plt.xlabel('Loss');
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/sep_loss_epoch%03d.png' % (path,epoch))
    plt.clf()  

def plot_histogram_pred(data, inds_clean, inds_noisy, path, epoch):

    bins = compute_histogram_bins(data, 0.01)

    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

    plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
    if len(inds_noisy) >0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
    plt.xlabel('Prob');
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/preds_sep_epoch%03d.png' % (path,epoch))
    plt.clf() 




def plot_model_view_histogram_loss(data, idx_view_labeled, idx_view_unlabeled, inds_clean, inds_noisy, path, epoch):
    
    bins = compute_histogram_bins(data, 0.01)
    #plot alg sep loss view

    num_view_labeled = len(idx_view_labeled)
    num_view_unlabeled = len(idx_view_unlabeled)
    total = num_view_labeled + num_view_unlabeled

    missed_clean = np.asarray([i for i in inds_clean if i not in idx_view_labeled])
    missed_noisy = np.asarray([i for i in inds_noisy if i not in idx_view_unlabeled])
    
    plt.hist(data[idx_view_labeled],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_clean(%d| %.1f%%)'%(num_view_labeled,100*num_view_labeled/float(total)))
    plt.hist(data[idx_view_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_noisy(%d| %.1f%%)'%(num_view_unlabeled,100*num_view_unlabeled/float(total)))
    plt.hist(data[missed_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='#fb8072', label='FN (%d| %.1f%%)'%(len(missed_clean),100*len(missed_clean)/float(len(inds_clean))))
    if len(inds_noisy) >0:
        plt.hist(data[missed_noisy],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='k',label='FP (%d| %.1f%%)'%(len(missed_noisy),100*len(missed_noisy)/float(len(inds_noisy))))
    plt.xlabel('Loss');
    plt.ylabel('Number of data')
    plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/view_sep_loss_epoch%03d.png' % (path,epoch))
    plt.clf()  

def plot_model_view_histogram_pred(data, idx_view_labeled, idx_view_unlabeled, inds_clean, inds_noisy, path, epoch):
    bins = compute_histogram_bins(data, 0.01)

    #plot alg sep pred view
    num_view_labeled = len(idx_view_labeled)
    num_view_unlabeled = len(idx_view_unlabeled)
    total = num_view_labeled + num_view_unlabeled

    missed_clean = np.asarray([i for i in inds_clean if i not in idx_view_labeled])
    missed_noisy = np.asarray([i for i in inds_noisy if i not in idx_view_unlabeled])

    plt.hist(data[idx_view_labeled],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_clean(%d| %.1f%%)'%(num_view_labeled,100*num_view_labeled/float(total)))
    plt.hist(data[idx_view_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_noisy(%d| %.1f%%)'%(num_view_unlabeled,100*num_view_unlabeled/float(total)))
    plt.hist(data[missed_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='#fb8072', label='FN (%d| %.1f%%)'%(len(missed_clean),100*len(missed_clean)/float(len(inds_clean))))
    if len(inds_noisy) >0:
        plt.hist(data[missed_noisy],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='k',label='FP (%d| %.1f%%)'%(len(missed_noisy),100*len(missed_noisy)/float(len(inds_noisy))))
    plt.xlabel('Pred');
    plt.ylabel('Number of data')
    plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/view_sep_prob_epoch%03d.png' % (path,epoch))
    plt.clf()  

def plot_tpr_fpr(noisy_labels, clean_labels, prob ):
    clean = (np.array(noisy_labels)==np.array(clean_labels))
    fpr, tpr, threshold = metrics.roc_curve(clean, prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.clf()  
    
    
    tpr = (sum(clean)-len(missed_clean))/sum(clean)
    fpr = len(missed_noisy)/(len(clean)-sum(clean))
    # writer_tensorboard.add_scalar('Metrics/auc', roc_auc, epoch)
    # writer_tensorboard.add_scalar('Metrics/tpr', tpr, epoch)
    # writer_tensorboard.add_scalar('Metrics/fpr', fpr, epoch)



    



