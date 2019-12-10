import copy
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline

def get_warp_idx(rect, p):
    x,y,w,h = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
    t1 = np.linspace(x,x+w-1,np.around(w))
    t2 = np.linspace(y,y+h-1,np.around(h))
    x_idx, y_idx = np.meshgrid(t1,t2)
    idx_final = np.array([x_idx.flatten(), y_idx.flatten(), np.ones_like(x_idx.flatten())])
    p0_init = np.array([[1,0,p[1]],[0,1,p[0]]]).squeeze()
    warped_idx = np.matmul(p0_init, idx_final)
    return warped_idx


def get_interpolated_rect(mesh, warped_idx, dx, dy):
    # r, c = img.shape
    # mesh = RectBivariateSpline(np.linspace(0,r-1,r), np.linspace(0,c-1,c), img)
    interpolated_rect_img = mesh.ev(warped_idx[1], warped_idx[0], dx, dy)
    return interpolated_rect_img

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]
    # print (It.shape, It1.shape, rect, p0.shape)
    p_final = copy.deepcopy(p0)
    threshold = 5e-2
    r, c = It1.shape
    mesh = RectBivariateSpline(np.linspace(0,r-1,r), np.linspace(0,c-1,c), It1)
    while True:
        warped_idx = get_warp_idx(rect, p_final)
        interpolated_rect_x = get_interpolated_rect(mesh, warped_idx, 1, 0)
        interpolated_rect_y = get_interpolated_rect(mesh, warped_idx, 0, 1)
        interpolated_rect_img = get_interpolated_rect(mesh, warped_idx, 0, 0)
        A = np.vstack((interpolated_rect_x, interpolated_rect_y)).T
        b = It.flatten() - interpolated_rect_img
        p = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        # print (p)
        p_final += p
        if abs(p.mean()) < threshold:
            break
    # print ('-------')
    return p_final
