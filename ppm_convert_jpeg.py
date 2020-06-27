
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys


def read_training_subset(root_path, jpeg_root_path):

    images = []  # images
    labels = []  # corresponding labels
    num_total_img_train = 0
    # loop over all 42 classes.
    for c in range(0, 43):
        print('processing class {} images...'.format(c))
        prefix = os.path.join(root_path, format(c, '05d'))  # subdirectory for class.
        jpeg_prefix = os.path.join(jpeg_root_path, format(c, '05d'))

        anno_path = os.path.join(prefix, ('GT-' + format(c, '05d') + '.csv'))
        with open(anno_path) as gt_file:
            gt_reader = csv.reader(gt_file)  # csv parser for annotations file.

            # loop over all images in current annotations file.
            count = 0
            for row in gt_reader:
                if count == 0:
                    pass
                else:
                    img_name = row[0].split(';')[0]
                    img = plt.imread(os.path.join(prefix, img_name))

                    # save ppm format images to jpeg images
                    if not os.path.exists(jpeg_prefix):
                        os.makedirs(jpeg_prefix)
                    jpeg_img_name = img_name.replace('ppm', 'jpeg')
                    jpeg_img_path = os.path.join(jpeg_prefix, jpeg_img_name)

                    images.append(img)  # the 1th column is the image filename.
                    labels.append(row[0].split(';')[7])  # the 8th column is the label.
                count += 1
        num_total_img_train += count
        gt_file.close()

    print('number of all training imgs:', num_total_img_train)
    return images, labels

def read_test_subset(root_path, jpeg_root_path):
    num_total_img_test = 0
    for c in range(0, 43):
        print('processing class {} images...'.format(c))
        prefix = os.path.join(root_path, format(c, '05d'))  # subdirectory for class.
        jpeg_prefix = os.path.join(jpeg_root_path, format(c, '05d'))
        anno_path = os.path.join(prefix, ('GT-' + format(c, '05d') + '.csv'))
        with open(anno_path) as gt_file:
            gt_reader = csv.reader(gt_file)

            count = 0
            for row in gt_reader:
                if count == 0:
                    pass
                else:
                    img_name = row[0].split(';')[0]
                    img = plt.imread(os.path.join(prefix, img_name))

                    # save ppm format images to jpeg images
                    if not os.path.exists(jpeg_prefix):
                        os.makedirs(jpeg_prefix)
                    jpeg_img_name = img_name.replace('ppm', 'jpeg')
                    jpeg_img_path = os.path.join(jpeg_prefix, jpeg_img_name)
                    cv2.imwrite(jpeg_img_path, img)

                count += 1
        num_total_img_test += count

        gt_file.close()
    print('number of all training imgs:', num_total_img_test)


def main(argv):
    Cnn_Path=argv[1]

    read_test_subset(Cnn_Path,
     Cnn_Path+'/JPEGImages')


if __name__ == '__main__':

    main(sys.argv)
