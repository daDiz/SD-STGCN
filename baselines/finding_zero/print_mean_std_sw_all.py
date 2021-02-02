import numpy as np
import sys

file_name = sys.argv[1]
ws_id = int(sys.argv[2])

if ws_id == 0:
    ws = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
else:
    ws = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8]

def calc_mean_std(file_name, w):
    scores = []
    with open(file_name, 'r') as file:
        file.readline()
        for line in file:
            x = np.array(line.strip().split(' '))
            dat = [float(a) for a in x]
            if dat[0] == w:
                scores.append(dat[1:])

    scores = np.array(scores)

    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    return mean_scores, std_scores

def print_scores(mean_scores, std_scores, window):
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (window, mean_scores[0],mean_scores[1],mean_scores[2],mean_scores[3]))
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (window, std_scores[0],std_scores[1],std_scores[2],std_scores[3]))

print('Pct\tAcc\tMrr\tHit5\tHit10')
for w in ws:
    m, s = calc_mean_std(file_name, w)
    print_scores(m, s, w)


