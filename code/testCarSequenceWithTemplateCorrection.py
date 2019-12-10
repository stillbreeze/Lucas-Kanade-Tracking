import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline

from LucasKanade import LucasKanade, get_warp_idx, get_interpolated_rect

# write your script here, we recommend the above libraries for making your animation

if __name__ == '__main__':
    car_video = np.load('../data/carseq.npy')
    frame_count = car_video.shape[-1]
    a,b,c,d = 59, 116, 145, 151
    p = np.zeros(2)
    It = car_video[b:d,a:c,0]
    rect = np.array([a,b,c,d])
    first_It = car_video[b:d,a:c,0]
    first_rect = np.array([a,b,c,d])
    epsilon = 2
    all_rect = []
    all_rect.append(rect)

    for t in range(1,frame_count):
        print (t)
        It1 = car_video[:,:,t]
        rows, columns = It1.shape
        mesh = RectBivariateSpline(np.linspace(0,rows-1,rows), np.linspace(0,columns-1,columns), It1)
        p = LucasKanade(It, It1, rect)
        p_star = LucasKanade(first_It, It1, first_rect)

        if np.linalg.norm(p_star - p, 1) < epsilon:
            # print ('updating template')
            warped_idx = get_warp_idx(rect, p)
            It = get_interpolated_rect(mesh, warped_idx, 0, 0)
            It = It.reshape(d-b, c-a)
        rect = np.array([rect[0]+p[1],rect[1]+p[0],rect[2]+p[1],rect[3]+p[0]])
        all_rect.append(rect)
        visualise_img = cv2.rectangle(It1.copy(), (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), (0,0,0), 1)
        cv2.imshow('Template correction LK', visualise_img)
        cv2.waitKey(1)
        # if t % 100 == 0 or t==1:
        #     fig,ax = plt.subplots(1)
        #     rect_bbox = patches.Rectangle((rect[0],rect[1]),rect[2] - rect[0],rect[3] - rect[1],linewidth=1,edgecolor='r',facecolor='none')
        #     ax.add_patch(rect_bbox)
        #     ax.imshow(It1, cmap=plt.gray())
        #     # plt.show()
        #     plt.savefig('../results/corrected_'+str(t)+'.png')
        #     plt.close()
    all_rect = np.asarray(all_rect)
    np.save('../results/carseqrects-wcrt.npy', all_rect)
    cv2.destroyAllWindows()



