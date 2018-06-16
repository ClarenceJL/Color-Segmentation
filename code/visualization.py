"""
Author: Claire Li
Date: Jan, 2018
Reference: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
"""

import numpy as np
import cv2
import colorsys
import matplotlib.image
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D



def visualize_segmentation(X, labels, batch_size, savefile=False, filename='', show=False, height=900, width=1200):
    """
    visualize the segmentation result
    opt: draw bounding boxes
    :param X: [n, 3]
    :param labels: [n,], boolean
    :param batch_size: scalar
    :param height: scalar
    :param width: scalar
    :return:
    """
    # reshape data
    imgs = np.reshape(X, [batch_size, height, width, 3])
    masks = np.reshape(labels, [batch_size, height, width])
    fig = plt.figure()
    for i in range(batch_size):
        plt.subplot(2, batch_size, i+1)
        # change color space
        img = cv2.cvtColor(imgs[i], cv2.COLOR_HSV2RGB)
        plt.imshow(img)

        plt.subplot(2, batch_size, i+1+batch_size)
        # overlay segmentation mask on it
        segmask = np.zeros([height, width, 3], dtype=np.uint8)
        segmask = apply_mask(segmask, masks[i])
        plt.imshow(segmask)

    if savefile:
        fig.savefig(filename+'.png')

    if show:
        plt.show()


def visualize_proposals(mask, img, bboxes, centers, depths, savefile=False, filename='_', show=False):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    fig, ax = plt.subplots(1)
    img_height, img_width = img.shape[:2]
    ax.set_ylim(img_height + 10, -10)
    ax.set_xlim(-10, img_width + 10)
    ax.axis('off')
    ax.set_title(filename)

    num_proposals = np.shape(bboxes)[0]
    segmask = np.tile(np.reshape(mask.astype(np.uint8)*255, [img_height, img_width, 1]), [1, 1, 3])
    segmask[:, :, 0] = 0

    for i in range(num_proposals):
        # draw bounding box
        minx, miny, w, h = bboxes[i]
        center = centers[i]
        bb = patches.Rectangle((minx, miny), w, h, alpha=0.5, linewidth=1,
                               edgecolor='r', facecolor='none')
        ax.add_patch(bb)

        # write label
        cap_d = 'Depth: {0:.2f}'.format(float(depths[i]))
        cap_c = 'Center:({0:.2f}, {1:.2f})'.format(center[1], center[0])
        ax.text(minx, miny - 8, cap_d, color='w', size=8, backgroundcolor="none")
        ax.text(minx, miny - 32, cap_c, color='w', size=8, backgroundcolor="none")

        print('{0:s} & {1:.4f} & {2:.4f} & {3:.4f}'.format(filename[:-4], center[1], center[0], float(depths[i])))

    ax.imshow(img)
    ax.imshow(segmask, alpha=0.5)


    if savefile:
        fig.savefig(filename+'.png')

    if show:
        plt.show()


def plot_histogram(y, ymin=0.0, ymax=1.0, num_bins=100):
    """
    plot histogram of data
    :param y: [n,] 1D vector
    :param ymin: minimum value
    :param ymax: maximum value
    :param num_bins: number of bins
    :return:
    """
    plt.hist(y, range=(ymin, ymax), bins=num_bins)
    plt.title("histogram of distribution")
    plt.show()


def apply_mask(image, mask, color=[0, 1, 1], alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == True,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors
