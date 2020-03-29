import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


obs_cols = ['frame_number', "pedestrian_ID", "pos_x", "pos_y"]
url = '/home/want/Project/SoTrajectory/datasets/zara1/train/students001_train.txt'
df = pd.read_csv(url, delimiter='\t', names = obs_cols)
groupby_report = df.groupby(['pedestrian_ID']).size().reset_index(name='counts')
threshold_frames = int(groupby_report.counts.quantile(0.25))
groupby_report_cropped = groupby_report[groupby_report.counts>=threshold_frames]
data_clean = df[df.pedestrian_ID.isin(groupby_report_cropped.pedestrian_ID)]
traj_x = []
traj_y = []
indexs = data_clean['pedestrian_ID'].unique()

for i in indexs:
    traj_x.append(df[df.pedestrian_ID.isin([i])]['pos_x'].values)
    traj_y.append(df[df.pedestrian_ID.isin([i])]['pos_y'].values)

fig = plt.figure()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame

traj_len = 16
peds_num = 10

x_start_end = [min(traj_x[:][0]), max(traj_x[:][traj_len-1])]
y_start_end = [min(traj_y[:][0]), max(traj_y[:][traj_len-1])]
plt.xlim(x_start_end)
plt.ylim(y_start_end)

ims = []
for t in range(traj_len):
    im = []
    for i in range(peds_num):
        im.append(plt.scatter(traj_x[i][:t], traj_y[i][:t]))
    ims.append(im)


# for i in range(16):
#     im = plt.scatter(traj_x[5][:i], traj_y[5][:i])
#     ims.append([im])
#
# print((traj_x[5][:20], traj_y[5][:20]))

ani = animation.ArtistAnimation(fig, ims, interval=500)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.show()
