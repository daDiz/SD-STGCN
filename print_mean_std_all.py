import numpy as np
import sys

R0 = sys.argv[1]
dataName = sys.argv[2]

if dataName == 'ER':
    prefix = 'ER-SIR-1000-%s-14-' % (R0)
elif dataName == 'bkFratB':
    prefix = 'bkFratB-SIR-58-%s-14-' % (R0)
elif dataName == 'highSchool':
    prefix = 'highSchool-SIR-774-%s-14-' % (R0)
elif dataName == 'sfhh':
    prefix = 'sfhh-SIR-403-%s-14-' % (R0)

if dataName == 'ER':
    ws = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
else:
    ws = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8]



def calc_mean_std(file_name):
    scores = []
    with open(file_name, 'r') as file:
        file.readline()
        for line in file:
            x = np.array(line.strip().split(' '))
            dat = [float(a) for a in x]
            scores.append(dat)

    scores = np.array(scores)

    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    return mean_scores, std_scores

def print_scores(mean_scores, std_scores, window):
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (window, mean_scores[0],mean_scores[1],mean_scores[2],mean_scores[3]))
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (window, std_scores[0],std_scores[1],std_scores[2],std_scores[3]))




print('Pct\tAcc\tMrr\tHit5\tHit10')
for w in ws:
    fileName = prefix + str(w) +'.dat'
    m, s = calc_mean_std(fileName)
    print_scores(m, s, w)




