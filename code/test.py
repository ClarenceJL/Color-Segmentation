import cv2
import os
import numpy as np
import pickle
from barrel_localization import region_proposals, DistanceRegression
from GMM import GaussianMixtureModel
from visualization import visualize_proposals

folder = "../test_data/"
#folder = "../train_data/"

# load regression model
c_model = pickle.load(open('color_seg_model', 'rb'))
# load depth estimation model
d_model = pickle.load(open('depth_est_model', 'rb'))

image_names = [p for p in os.listdir(folder) if p.endswith('.png')]

for filename in image_names:
    # read one test image
    bgr_img = cv2.imread(os.path.join(folder, filename))
    h, w, d = np.shape(bgr_img)
    # Your computations here!
    # pre-process test data
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)# change color space
    pixels = np.reshape(hsv_img, [-1, 3])
    prob_est, labels_est = c_model.inference(pixels)
    raw_mask = np.reshape(labels_est, [h, w])

    # do barrel localization and distance estimation
    new_mask, bboxes, centers, depths = region_proposals(raw_mask, d_model)
    
    # Display results:
    # (1) Segmented image
    # (2) Barrel bounding box
    # (3) Distance of barrel
    visualize_proposals(new_mask, hsv_img, bboxes, centers, depths,
                        savefile=True, filename=filename[:-4]+'_res', show=False)
    
    # You may also want to plot and display other diagnostic information cv2.waitKey(0)
    cv2.destroyAllWindows()