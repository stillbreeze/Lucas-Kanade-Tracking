import copy
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform, sobel

def get_warp_idx(It, p):
    M = np.array([[1+p[0], p[1], p[2]], [p[3],1+p[4],p[5]], [0, 0, 1]]).squeeze()
    blank = np.ones_like(It)
    warped_blank = affine_transform(blank, M)
    return warped_blank > 0.0, M

def get_warped_region(img, warped_idx_bool, M):
    warped_img = affine_transform(img, M)
    common_warped_img = warped_img[warped_idx_bool]
    return common_warped_img

def InverseCompositionAffine(It, It1):
    # Input: 
    #   It: template image
    #   It1: Current image
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    p0 = np.zeros(6)
    # threshold = 5e-2
    threshold = 0.01

    Itx = sobel(It, axis=0, mode='constant')
    Ity = sobel(It, axis=1, mode='constant')

    warped_idx_bool, M = get_warp_idx(It1, p0)

    interpolated_rect_x = get_warped_region(Itx, warped_idx_bool, M)
    interpolated_rect_y = get_warped_region(Ity, warped_idx_bool, M)

    x_values, y_values = np.where(warped_idx_bool != False)

    a1 = interpolated_rect_x * x_values
    a2 = interpolated_rect_x * y_values
    a3 = interpolated_rect_x
    a4 = interpolated_rect_y * x_values
    a5 = interpolated_rect_y * y_values
    a6 = interpolated_rect_y
    A = np.column_stack((a1,a2,a3,a4,a5,a6))

    while True:
        interpolated_rect_img = get_warped_region(It1, warped_idx_bool, M)
        b = interpolated_rect_img - It[warped_idx_bool].flatten()
        p = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        del_M = np.array([[1+p[0], p[1], p[2]], [p[3],1+p[4],p[5]], [0, 0, 1]]).squeeze()
        M = M.dot(np.linalg.inv(del_M))
        if abs(p.mean()) < threshold:
            break    
    return M

