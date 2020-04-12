import random
import matplotlib.pyplot as plt

length = 16 # length of traj
obs_len = 8
pre_len = 8
person_num = 300  # number of traj
straight = [10 for i in range(obs_len)]
left = [(10 + i**2 * 0.1) for i in range(obs_len)]
right = [(10 - i**2 * 0.1) for i in range(obs_len)]
base_straight = straight + straight
base_left = straight + left
base_right = straight + right
filename = './toydata/test/left.txt'
file = open(filename, mode='w+')
x = [i for i in range(length)]
y = base_left
for idx in range(person_num):
    for i in range(length):
        file.writelines([str((i+1)*10), '\t', str(idx+1), '\t', str(x[i]+random.random()), '\t',str(y[i]+random.random())])  # frame_id    person_id   x    y
        file.write('\n')

file.close()
