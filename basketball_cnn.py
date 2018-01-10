from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nprand
import tensorflow as tf
import cv2 as cv
import argparse
import sys
import tempfile
import csv
import glob
import pandas as pd
import random
from tensorflow.contrib.data import Dataset, Iterator

IMAGE_DIR = r'./processing_v2'

def get_list_file_annotations(folder):
    return glob.glob(folder + '/*.annote');

def get_image_training_data(img_tuple):
    """ take the image tuple"""
    img = cv.imread(img_tuple[0], 1) # i
    (window_x, window_y) = img_tuple[1]
    windows = img_tuple[2]
    data = np.ones((len(windows), window_x * window_y * 3))
    ground_truth = np.zeros((len(windows), 2));
    i = 0
    for window in windows:
        x_pos = int(window[0])
        y_pos = int(window[1])
        is_player = int(window[2]) == 1
        window_image = img[y_pos:y_pos + window_y, x_pos:x_pos + window_x]
        data[i] = window_image.reshape(window_x * window_y * 3)
        if is_player:
            ground_truth[i][0] = 1
            ground_truth[i][1] = 0
        else:
            ground_truth[i][0] = 0
            ground_truth[i][1] = 1
        i = i + 1
    return (data, ground_truth)

def get_data():
    # list of annotation files

    files = get_list_file_annotations(IMAGE_DIR)
    
    training = []
    for file in files:
        # create a tuple representation of annotation
        with open(file, 'r') as f:
            reader = csv.reader(f)
            fileName = IMAGE_DIR + "/" + next(reader)[0]
            dims = next(reader)
            window_x = int(dims[0])
            window_y = int(dims[1])
            windows = []
            for row in reader:
                windows.append(row)
            # now
            (data, ground_truth) = get_image_training_data(
                (fileName, (window_x, window_y), windows)
            )
            for i in range(0, len(data)):
                training.append((data[i], ground_truth[i]))
    training_images = np.zeros((len(training), 40000 * 3))
    training_ground_truth = np.zeros((len(training), 2))

    for i in range(0, len(training)):
        training_images[i] = training[i][0]
        training_ground_truth[i] = training[i][1]
    return (training_images, training_ground_truth)

def get_rescaled_image(h,w):
    (images, ground_truth) = get_data()
    images_down_scaled = np.zeros((len(images), h * w * 3))

    for i in range(0, images.shape[0]):
        img = images[i].reshape(200,200,3).astype(np.uint8)
        res = cv.resize(img,(h, w), interpolation = cv.INTER_CUBIC)
        images_down_scaled[i] = res.reshape(h*w*3)
    return (images_down_scaled, ground_truth)

tf.set_random_seed(1234)
h = 200
w = 200
minibatch_size = 1
FILTERS = 10
KERNEL_SIZE = [4, 4]
RELU = tf.nn.relu
SOFTMAX = tf.nn.softmax
POOL_SIZE = [2, 2]
POOL_STRIDE = 2
OUTPUT_CLASS = 2
LEARNING_RATE = 0.01
TOTAL_ITERATIONS = 1000
TEST_CASES = 55 # number of test cases. not to be confused with test_cases, which is the array
# containing the indices of those test cases

# separating data for testing

test_cases = random.sample(range(0, 455), TEST_CASES)

# preparing the input data

[training_images, training_labels] = get_rescaled_image(h, w)

my_image = cv.imread(IMAGE_DIR+'/DSCF2394.annote.PNG', 1) # for verification purposes

input_data = []
label_data = []
temp = [] # for each minibatch. this will be appended to input_data []
temp_labels = []
count = 0
for window in range(0, 455):
    if window in test_cases:
        continue
    temp.append(training_images[window].reshape(h, w, 3))
    temp_labels.append(training_labels[window])
    count += 1
    if count == minibatch_size:
        input_data.append(temp)
        label_data.append(temp_labels)
        temp = []
        temp_labels = []
        count = 0

# preparing the test data

test_input = []
test_label = []
for index in range(TEST_CASES):
    test_input.append(training_images[test_cases[index]].reshape(1, h, w, 3))
    test_label.append(training_labels[test_cases[index]].reshape(1, OUTPUT_CLASS))

# CNN model

input_layer = tf.placeholder(tf.float32, [minibatch_size, h, w, 3]) # input layer
true_label = tf.placeholder(tf.float32, [minibatch_size, OUTPUT_CLASS]) # input layer

# first conv layer and pooling
conv11 = tf.layers.conv2d(
    inputs= input_layer,
    filters= FILTERS,
    kernel_size= KERNEL_SIZE,
    padding= 'same',
    activation = RELU)
conv12 = tf.layers.conv2d(
    inputs= conv11,
    filters= FILTERS,
    kernel_size= KERNEL_SIZE,
    padding= 'same',
    activation = RELU)
pool1 = tf.layers.max_pooling2d(
    inputs= conv12,
    pool_size= POOL_SIZE,
    strides= POOL_STRIDE)

#second conv layer and pooling
conv21 = tf.layers.conv2d(
    inputs= pool1,
    filters= 2*FILTERS,
    kernel_size= KERNEL_SIZE,
    padding= 'same',
    activation = RELU)
conv22 = tf.layers.conv2d(
    inputs= conv21,
    filters= 2*FILTERS,
    kernel_size= KERNEL_SIZE,
    padding= 'same',
    activation = RELU)
pool2 = tf.layers.max_pooling2d(
    inputs= conv22,
    pool_size= POOL_SIZE,
    strides= POOL_STRIDE)

# dense network
pool2_flat = tf.reshape(pool2, [-1, int((h/(POOL_SIZE[0]**2))*(w/(POOL_SIZE[1]**2))*2*FILTERS)])
dense1 = tf.layers.dense(
    inputs= pool2_flat,
    units= 200,
    activation= SOFTMAX)
dense2 = tf.layers.dense(
    inputs= dense1,
    units= OUTPUT_CLASS,
    activation= SOFTMAX)

# updating the values
error = -tf.reduce_mean(true_label*tf.log(dense2))
train_cnn = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error)

# training the CNN model

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("Before training")

_ = 0
for _ in range(TOTAL_ITERATIONS):
    correct = 0
    for inp, lab in zip(test_input, test_label):
        result = sess.run(dense2, feed_dict={input_layer: inp, true_label: lab})
        pred = np.sum(result * np.asarray(lab), axis=1)[0]
        correct += int(pred/0.5)
    print("Accuracy = ", correct/float(TEST_CASES), _)
    tr_ex = 1
    for [batch_image, batch_label] in zip(input_data, label_data):
        training = sess.run(train_cnn, feed_dict={input_layer: batch_image, true_label: batch_label})
        train_error = sess.run(error, feed_dict={input_layer: batch_image, true_label: batch_label})
        print("Error ", train_error, "Epoch ", _, "Training example ", tr_ex)
        tr_ex = tr_ex+1
        print(sess.run(dense2, feed_dict={input_layer: batch_image, true_label: batch_label}))
    _ = _+1
    # print(sess.run(dense2, feed_dict={input_layer: batch_image, true_label: batch_label}))
    # testing the CNN model

correct = 0
print("After training")
for inp, lab in zip(test_input, test_label):
    result = sess.run(dense2, feed_dict={input_layer: inp, true_label: lab})
    pred = np.sum(result * np.asarray(lab), axis=1)[0]
    correct += int(pred/0.5)
print("Accuracy = ", correct/float(TEST_CASES))
