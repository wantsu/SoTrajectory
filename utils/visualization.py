import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


# (len, batch, 2)
def visualization(traj_gt, traj_pred, obs_len):
    traj_len = traj_gt.size(0)
    peds_num = traj_gt.size(1)
    traj_gt_x, traj_pred_x, traj_gt_y, traj_pred_y= [], [], [], []
    if peds_num >= 8:
        peds_num = 8

    for i in range(peds_num):
        traj_gt_x.append(traj_gt[::,i][:,0].numpy())
        traj_gt_y.append(traj_gt[::,i][:,1].numpy())
        traj_pred_x.append(traj_pred[::,i][:,0].detach().numpy())
        traj_pred_y.append(traj_pred[::,i][:,1].detach().numpy())

    _animation(peds_num, traj_len, obs_len, traj_gt_x, traj_gt_y, traj_pred_x, traj_pred_y)


# plot as an animation
def _animation(peds_num, traj_len, obs_len, traj_gt_x, traj_gt_y, traj_pred_x, traj_pred_y):

    x_start_end = [min([min(x) for x in traj_gt_x]), max([max(x) for x in traj_gt_x])]
    y_start_end = [min([min(y) for y in traj_gt_y]), max([max(x) for x in traj_gt_y])]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('Ground truth')
    ax2.set_title('Prediction')
    plt.xlim(x_start_end)
    plt.ylim(y_start_end)
    ims = []
    for t in range(traj_len):
        im = []
        for i in range(peds_num):
            im.append(ax1.scatter(traj_gt_x[i][:t], traj_gt_y[i][:t], c='g'))
            im.append(ax1.scatter(traj_pred_x[i][:t], traj_pred_y[i][:t], c='r'))
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=500)
    #ani.save(time.strftime('Img/'+'%Y-%m-%d %H:%M:%S'.format(time.localtime(time.time()))+'.Gif' ))
    plt.show()

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # from matplotlib.animation import FFMpegWriter
    # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)



