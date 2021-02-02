import numpy as np
import pickle
from metric_utils import *
import sys

T = int(sys.argv[1])

n_files = 10
nn = 100

def load_pred(T,n_files,nn,prefix='dmp_pred'):
    '''
    T: time of the snapshot
    n_files: num of files to stich together
    nn: num of nodes in a file
    '''
    data = None
    for i in range(n_files):
        start = i * nn
        end = (i+1) * nn
        f = '%s_%s-%s-%s.pickle' % (prefix, T, start, end)
        x = np.asarray(pickle.load(open(f, 'rb')).todense())
        if i == 0:
            data = x
        else:
            data = np.concatenate((data, x), axis=1)

    return data

#print(pred.shape)

train_pct = 0.8
val_pct = 0.1


# load the labels (sources) in the test set
def load_label(file_name):
    _, y = pickle.load(open(file_name, 'rb'))
    n_tot = len(y)
    n_train = int(n_tot * train_pct)
    n_val = int(n_tot * val_pct)

    y_test = np.array(y)[n_train+n_val:]

    return y_test

acc_tot = []
mrr_tot = []
hit5_tot = []
hit10_tot = []
hit20_tot = []
for i in range(5):
    prefix = './results/SIR-ER-1000-T30-g%s/dmp_pred' % (i)

    pred = load_pred(T, n_files, nn, prefix)

    label_file =\
    '../../dataset/ER/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_p0.02_g%s_entire.pickle'\
    % (i)

    y_test = load_label(label_file) # this load the test set label

    acc_tot.append(np.mean(acc(pred, y_test)))
    mrr_tot.append(np.mean(mrr(pred, y_test)))
    hit5_tot.append(np.mean(hit_at(pred, y_test, 5)))
    hit10_tot.append(np.mean(hit_at(pred, y_test, 10)))
    hit20_tot.append(np.mean(hit_at(pred, y_test, 20)))

mean_acc = np.mean(acc_tot)
mean_mrr = np.mean(mrr_tot)
mean_hit5 = np.mean(hit5_tot)
mean_hit10 = np.mean(hit10_tot)
mean_hit20 = np.mean(hit20_tot)

std_acc = np.std(acc_tot)
std_mrr = np.std(mrr_tot)
std_hit5 = np.std(hit5_tot)
std_hit10 = np.std(hit10_tot)
std_hit20 = np.std(hit20_tot)


print(f'Acc: {mean_acc:.5f} +/- {std_acc:.7f}')
print(f'MRR: {mean_mrr:.5f} +/- {std_mrr:.7f}')
print(f'Hit@5: {mean_hit5:.5f} +/- {std_hit5:.7f}')
print(f'Hit@10: {mean_hit10:.5f} +/- {std_hit10:.7f}')
print(f'Hit@20: {mean_hit20:.5f} +/- {std_hit20:.7f}')




