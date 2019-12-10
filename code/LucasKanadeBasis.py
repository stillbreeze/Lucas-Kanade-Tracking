import copy
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline

from LucasKanade import get_warp_idx, get_interpolated_rect

def LucasKanadeBasis(It, It1, rect, bases):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   bases: [n, m, k] where nxm is the size of the template.
    # Output:
    #   p: movement vector [dp_x, dp_y]

    p0 = np.zeros(2)
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
        new_A = A - bases.dot(bases.T).dot(A)
        new_B = b - bases.dot(bases.T).dot(b)
        # print (new_A.shape, new_B.shape)
        p = np.linalg.inv(new_A.T.dot(new_A)).dot(new_A.T).dot(new_B)
        # print (p)
        p_final += p
        if abs(p.mean()) < threshold:
            break
    # print ('-------')
    return p_final
    
