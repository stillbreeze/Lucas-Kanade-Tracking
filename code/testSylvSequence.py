import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline

from LucasKanade import get_warp_idx, get_interpolated_rect, LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

# write your script here, we recommend the above libraries for making your animation


if __name__ == '__main__':
    car_video = np.load('../data/sylvseq.npy')
    bases = np.load('../data/sylvbases.npy')

    bases = np.swapaxes(bases, 0, 1)
    bases = bases.reshape(bases.shape[0]*bases.shape[1], bases.shape[2])

    frame_count = car_video.shape[-1]
    # a,b,c,d = 101, 61, 155, 107
    a,b,c,d = 101, 61, 156, 108
    It_bases = car_video[b:d,a:c,0]
    It_orig = car_video[b:d,a:c,0]
    rect_bases = np.array([a,b,c,d])
    rect_orig = np.array([a,b,c,d])
    all_rect = []
    all_rect.append(rect_bases)

    for t in range(1,frame_count):
        print (t)
        It1 = car_video[:,:,t]
        rows, columns = It1.shape
        mesh = RectBivariateSpline(np.linspace(0,rows-1,rows), np.linspace(0,columns-1,columns), It1)
        p_bases = LucasKanadeBasis(It_bases, It1, rect_bases, bases)
        warped_idx = get_warp_idx(rect_bases, p_bases)
        It_bases = get_interpolated_rect(mesh, warped_idx, 0, 0)
        It_bases = It_bases.reshape(d-b, c-a)
        rect_bases = np.array([rect_bases[0]+p_bases[1],rect_bases[1]+p_bases[0],rect_bases[2]+p_bases[1],rect_bases[3]+p_bases[0]])

        p_orig = LucasKanade(It_orig, It1, rect_orig)
        warped_idx = get_warp_idx(rect_orig, p_orig)
        It_orig = get_interpolated_rect(mesh, warped_idx, 0, 0)
        It_orig = It_orig.reshape(d-b, c-a)
        rect_orig = np.array([rect_orig[0]+p_orig[1],rect_orig[1]+p_orig[0],rect_orig[2]+p_orig[1],rect_orig[3]+p_orig[0]])
        all_rect.append(rect_bases)
        visualise_img = cv2.rectangle(It1.copy(), (int(rect_orig[0]),int(rect_orig[1])), (int(rect_orig[2]),int(rect_orig[3])), (0,0,0), 1)
        visualise_img = cv2.rectangle(visualise_img.copy(), (int(rect_bases[0]),int(rect_bases[1])), (int(rect_bases[2]),int(rect_bases[3])), (255,255,255), 1)
        cv2.imshow('Appearance basis and regular LK', visualise_img)
        cv2.waitKey(1)
        # if t in [1, 200, 300, 350, 400]:
        #     fig,ax = plt.subplots(1)
        #     rect_bbox_orig = patches.Rectangle((rect_orig[0],rect_orig[1]),rect_orig[2] - rect_orig[0],rect_orig[3] - rect_orig[1],linewidth=1,edgecolor='g',facecolor='none')
        #     ax.add_patch(rect_bbox_orig)
        #     rect_bbox_bases = patches.Rectangle((rect_bases[0],rect_bases[1]),rect_bases[2] - rect_bases[0],rect_bases[3] - rect_bases[1],linewidth=1,edgecolor='y',facecolor='none')
        #     ax.add_patch(rect_bbox_bases)
        #     ax.imshow(It1, cmap=plt.gray())
        #     # plt.show()
        #     plt.savefig('../results/bases_'+str(t)+'.png')
        #     plt.close()
    all_rect = np.asarray(all_rect)
    np.save('../results/sylvseqrects.npy', all_rect)