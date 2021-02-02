import numpy as np
import sys

file_name = sys.argv[1]
w = float(sys.argv[2])

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

print('window at %s' % (w))
print('mean:')
print(mean_scores)
print('stdev:')
print(std_scores)
