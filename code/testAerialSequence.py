import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from scipy.ndimage import affine_transform, grey_dilation

from SubtractDominantMotion import SubtractDominantMotion


# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    car_video = np.load('../data/aerialseq.npy')
    frame_count = car_video.shape[-1]


    fig, ax = plt.subplots()
    def update(i):
        ax.imshow(car_video[:,:,i], cmap=plt.gray())
        ax.imshow(mask_list[i], cmap='jet', alpha=0.5)


    mask_list = []
    for t in range(1,frame_count):
        print (t)
        It1 = car_video[:,:,t]
        It = car_video[:,:,t-1]
        mask = SubtractDominantMotion(It1, It)
        fs = 7
        dilated_img = grey_dilation(mask, size=(fs,fs))
        mask_list.append(dilated_img)

        # if t % 30 == 0 and t <= 120:
        #     fs = 7
        #     dilated_img = grey_dilation(mask, size=(fs,fs))
        #     fig,ax = plt.subplots(1)
        #     ax.imshow(It1, cmap=plt.gray())
        #     ax.imshow(dilated_img, cmap='jet', alpha=0.5)
        #     # plt.show()
        #     plt.savefig('../results/dominant_new_'+str(t)+'.png')
        #     plt.close()
    anim = animation.FuncAnimation(fig, update, frames=frame_count-1, interval=5)
    plt.show()


