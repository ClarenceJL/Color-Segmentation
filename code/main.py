"""
Author: Claire Li
Date: Jan, 2018
"""
import os
import pickle
import numpy as np
import cv2
from random import shuffle
from GMM import GaussianMixtureModel
from MDGM import MD_Gaussian_Model
from visualization import visualize_segmentation, plot_histogram

def load_data(file_path, names):
    images = []
    labels = []
    for name in names:
        bgr_img = cv2.imread(file_path + name)
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        label = np.load(file_path +'labels/' + name[:-4] + '.npy')
        images.append(img)
        labels.append(label)

    return images, labels

def cross_validation(file_path, num_folds=10):
    """
    :return: accuracy
    """
    # load data
    # split into m-fold
    image_names = [p for p in os.listdir(file_path) if p.endswith('.png')]
    num_images = len(image_names)
    batch_size = int(num_images / num_folds)
    shuffle(image_names)
    images, labels = load_data(file_path, image_names)

    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    for i in range(1):
        print('cross validation round: {}'.format(i))
        # split data
        rb = min(num_images, (i + 1) * batch_size)
        batch_ind = range(i*batch_size, rb)
        X_train = np.delete(images, batch_ind, axis=0)
        l_train = np.delete(labels, batch_ind, axis=0)
        X_val = images[i*batch_size:rb]
        l_val = labels[i*batch_size:rb]

        # train model
        X_train = np.reshape(X_train, [-1, 3])
        l_train = np.reshape(l_train, [-1])
        X_train_pos = X_train[l_train,:]
        X_train_neg = X_train[~l_train, :]
        # shuffle training data by pixel
        shuffle(X_train_pos)
        #shuffle(X_train_neg)
        model = GaussianMixtureModel(X_train_pos, X_train_neg, K_red=5)
        #model = MD_Gaussian_Model(X_train_pos, X_train_neg)

        # validation
        X_val = np.reshape(X_val, [-1, 3])
        l_val = np.reshape(l_val, [-1])
        prob_est, labels_est = model.inference(X_val)

        #plot_histogram(prob_est)

        acc = np.mean((labels_est == l_val).astype(float))
        precision = np.mean((np.logical_and(labels_est, l_val)).astype(float)) / \
            np.mean(labels_est.astype(float))
        recall = np.mean((np.logical_and(labels_est, l_val)).astype(float)) / \
            np.mean(l_val.astype(float))
        avg_acc = avg_acc + acc
        avg_prec = avg_prec + precision
        avg_rec = avg_rec + recall
        print('accuracy: {}, precision: {}, recall: {}'.format(acc, precision, recall))

        visualize_segmentation(X_val, labels_est, batch_size, savefile=True, show=False, filename='val_'+str(i))

        # save validation data
        np.savez('val_result_'+str(i), data=np.reshape(X_val, [batch_size, 900, 1200, 3]),
                 mask=np.reshape(labels_est, [batch_size, 900, 1200]))

    avg_acc = avg_acc / num_folds
    avg_prec = avg_prec / num_folds
    avg_rec = avg_rec / num_folds
    print('average accuracy: {}, average precision: {}, average recall: {}'.format(avg_acc, avg_prec, avg_rec))



def train_color_segmentation_model(file_path):
    # train the model using all the data and save the model
    """
    data = np.concatenate((np.reshape(images,[-1,3]), np.reshape(labels.astype(), [-1])), axis=-1)
    shuffle(data)
    """
    image_names = [p for p in os.listdir(file_path) if p.endswith('.png')]
    shuffle(image_names)
    images, labels = load_data(file_path, image_names)
    X_train = np.reshape(images, [-1, 3])
    l_train = np.reshape(labels, [-1])

    X_train_pos = X_train[l_train, :]
    #shuffle(X_train_pos)
    X_train_neg = X_train[~l_train, :]

    # train and save model
    model = GaussianMixtureModel(X_train_pos, X_train_neg, K_red=5)
    pickle.dump(model, open('color_seg_model', 'wb'))

    # calculate training accuracy
    prob_est, labels_est = model.inference(X_train)
    acc = np.mean((labels_est == l_train).astype(float))
    precision = np.mean((np.logical_and(labels_est, l_train)).astype(float)) / \
                np.mean(labels_est.astype(float))
    recall = np.mean((np.logical_and(labels_est, l_train)).astype(float)) / \
                np.mean(l_train.astype(float))

    print('Training accuracy: {}, precision: {}, recall: {}'.format(acc, precision, recall))


if __name__ == "__main__":
    #cross_validation('../train_data/')
    train_color_segmentation_model('../train_data/')
