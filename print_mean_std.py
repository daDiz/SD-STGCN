import numpy as np
import sys

file_name = sys.argv[1]

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

print('mean:')
print(mean_scores)
print('stdev:')
print(std_scores)
