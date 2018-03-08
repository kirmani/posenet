#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import scipy
import sys
import traceback
import time
import tensorflow as tf

FILENAME = "final_run"
TF_TOPIC = '/tf'
IMAGE_TOPIC = '/hsrb/head_l_stereo_camera/image_raw'
MAP_FRAME = 'map'
ODOM_FRAME = 'odom'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(args):
    """ Main function. """
    # Open our rosbag.
    bag = rosbag.Bag(FILENAME + '.bag')

    # Create our TFRecordWriter.
    writer = tf.python_io.TFRecordWriter(FILENAME + '.tfrecords')

    # Keep track of our pose.
    pose = None

    for topic, msg, t in bag.read_messages():
        if (topic == TF_TOPIC and
                msg.transforms[0].header.frame_id == MAP_FRAME and
                msg.transforms[0].child_frame_id == ODOM_FRAME):
            pose = msg.transforms[0].transform
        if pose and topic == IMAGE_TOPIC:
            # print("found image")
            position = [
                pose.translation.x, pose.translation.y, pose.translation.z
            ]
            rotation = [
                pose.rotation.x, pose.rotation.y, pose.rotation.z,
                pose.rotation.w
            ]

            # scale = 256.0 / min(msg.height, msg.width)
            scale = 1.0
            height = int(msg.height * scale)
            width = int(msg.width * scale)
            img = scipy.misc.imresize(
                np.reshape(
                    np.frombuffer(msg.data, dtype=np.uint8),
                    [msg.height, msg.width, 3]), [height, width])
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # exit()

            # print(pose)
            print(position)
            # print(rotation)

            # Create a feature
            feature = {
                'train/position': _float_feature(position),
                'train/rotation': _float_feature(rotation),
                'train/width': _int64_feature(width),
                'train/height': _int64_feature(height),
                'train/image': _bytes_feature(
                    tf.compat.as_bytes(img.tostring()))
            }

            # Create an example protocol buffer
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

    # Close our TFRecordWriter.
    writer.close()
    sys.stdout.flush()

    # Close our bag.
    bag.close()


if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser(usage=globals()['__doc__'])
        parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help='verbose output')
        args = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if args.verbose:
            print(time.asctime())
        main(args)
        if args.verbose:
            print(time.asctime())
            print('TOTAL TIME IN MINUTES:',)
            print((time.time() - start_time) / 60.0)
        sys.exit(0)
    except KeyboardInterrupt as err:  # Ctrl-C
        raise err
    except SystemExit as err:  # sys.exit()
        raise err
    except Exception as err:
        print('ERROR, UNEXPECTED EXCEPTION')
        print(str(err))
        traceback.print_exc()
        sys.exit(1)
