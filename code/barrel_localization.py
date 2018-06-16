"""
Author: Claire Li
Date: Jan, 2018
"""
import pickle
import cv2
import numpy as np
from skimage import data, util
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from itertools import compress


##################### Region Proposal ######################

def inscribe_rect_size(H, W, theta):
    h = H
    w = W
    if (theta - np.pi/2) < 1e-8:
        return h, w
    k = (H*H-W*W) / ((H*H+W*W)*np.cos(theta*2))
    a = k - 1
    b = 2 * a * np.sin(2*theta)
    c = a + 1
    ts = np.roots([a, b, c])
    t = np.max(ts)
    if t < 0:
        return h, w
    h = np.sqrt((H*H-W*W) / ((t*t-1)*np.cos(theta*2)))
    w = t * h

    return h, w

def region_proposals(mask, depth_regressor):
    """
    :param mask: [h, w] boolean raw segmentation mask
    :param img: [h, w, 3] original image, in rgb color space
    :param depth_regressor: trained DistanceRegression object
    :return:
    """
    # pre-process the mask
    filtered = binary_erosion(mask.astype(int))
    dilated = binary_dilation(filtered, iterations=2)

    labels = label(dilated)

    #plt.imshow(labels)
    #plt.show()

    props = regionprops(labels)
    num_props = np.shape(props)[0]

    max_area = 0
    for i in range(num_props):
        rmin, cmin, rmax, cmax = props[i]['bbox']
        H = rmax - rmin + 1
        W = cmax - cmin + 1
        if H < 2.2 * W:
            max_area = max(max_area, props[i]['area'])

    new_mask = labels.copy()

    bboxes = []
    centers = []
    depths = []

    for i in range(num_props):
        prop = props[i]
        # shape and (perhaps) orientation
        ori = prop['orientation']
        major_axis = prop['major_axis_length']
        minor_axis = prop['minor_axis_length']
        rmin, cmin, rmax, cmax = props[i]['bbox']
        rc, cc = props[i]['centroid']
        H = rmax - rmin + 1
        W = cmax - cmin + 1

        area = prop['area']

        # filter by bbox shape
        if H > 2.2 * W or W > 2 * H:
            new_mask[rmin:rmax + 1, cmin:cmax + 1] = 0
            continue

        # filter by size,
        if prop['area'] < 500:
            new_mask[rmin:rmax + 1, cmin:cmax + 1] = 0
            continue


        if prop['extent'] < 0.45:
            new_mask[rmin:rmax + 1, cmin:cmax + 1] = 0
            continue

        if prop['area']/max_area < 0.4:
            new_mask[rmin:rmax + 1, cmin:cmax + 1] = 0
            continue

        # check aspect-ratio to remove other things
        # find h, w of the barrel using H, W and ori
        h, w = inscribe_rect_size(H, W, ori)
        # check aspect ratio
        if H < 2 * W and H > 0.8 * W:
            bboxes.append([cmin, rmin, W, H])
            centers.append([rc, cc])
            depths.append(depth_regressor.inference([major_axis * 0.85], [minor_axis * 0.85]))
        else:
            new_mask[rmin:rmax+1, cmin:cmax+1] = 0 # remove invalid object

    new_mask = new_mask > 0
    return new_mask, bboxes, centers, depths,


class DistanceRegression(object):
    def __init__(self, heights, widths, depths):
        # training data
        self.training_area = np.multiply(heights, widths)
        self.training_depths = depths
        # parameter
        self.alpha = 1

        self.regression()

    def regression(self):
        # MLE for model d = alpha * (1 / sqrt(area))
        x = 1/np.sqrt(self.training_area)
        y = self.training_depths
        self.alpha = np.sum(np.multiply(x, y)) / np.dot(x, x)

    def inference(self, w, h):
        X = np.sqrt(np.multiply(w, h))
        depths_est = np.array([self.alpha if x < 1 else self.alpha / x for x in X], dtype=float)
        return depths_est


if __name__=='__main__':
    """
    run distance regression model
    """
    # read data from text
    filename = '../train_data/scales.txt'
    with open(filename) as f:
        content = f.readlines()

    # parse each line
    widths = []
    heights = []
    depths = []
    for s in content:
        n, d, w, h = s.split()
        depths.append(float(d))
        widths.append(float(w))
        heights.append(float(h))


    depth_regressor = DistanceRegression(heights, widths, depths)

    average_aspect_ratio = np.mean(np.asarray(heights)/np.asarray(widths))
    print('average_aspect_ratio: {}'.format(average_aspect_ratio))

    #np.savez('sp_model', depth_reg=depth_regressor, ar=average_aspect_ratio)
    pickle.dump(depth_regressor, open('depth_est_model', 'wb'))









