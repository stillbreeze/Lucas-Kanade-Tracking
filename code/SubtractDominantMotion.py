import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    # threshold = 0.15
    threshold = 0.20
    M = InverseCompositionAffine(image1, image2)
    warped_image1 = affine_transform(image2, M)
    mask = np.abs(warped_image1 - image1) > threshold
    return mask
