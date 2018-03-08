#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
PoseNet
"""
import argparse
import numpy as np
import sys
import traceback
import tensorflow as tf
import time

CROP_SIZE = 224


def Parser(record):
    # Parse the TF record
    parsed = tf.parse_single_example(
        record,
        features={
            'train/height': tf.FixedLenFeature([], tf.int64),
            'train/width': tf.FixedLenFeature([], tf.int64),
            'train/image': tf.FixedLenFeature([], tf.string),
            'train/position': tf.FixedLenFeature([3], tf.float32),
            'train/rotation': tf.FixedLenFeature([4], tf.float32),
        })
    # Load the data and format it
    H = tf.cast(parsed['train/height'], tf.int32)
    W = tf.cast(parsed['train/width'], tf.int32)
    image = tf.reshape(
        tf.decode_raw(parsed["train/image"], tf.uint8), [H, W, 3])
    position = tf.reshape(parsed["train/position"], [3])
    rotation = tf.reshape(parsed["train/rotation"], [4])

    ## Data augmentation
    # Randomly crop image.
    image = tf.random_crop(image, [CROP_SIZE, CROP_SIZE, 3])

    return image, tf.concat([position, rotation], axis=-1)


def LoadDataset(tfrecord):
    # Load the dataset
    dataset = tf.data.TFRecordDataset(tfrecord)

    # Parse the tf record entries
    dataset = dataset.map(Parser, num_parallel_calls=8)
    dataset.prefetch(1024)

    # Shuffle the data, batch it and run this for multiple epochs
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    return dataset


def main(args):
    """ Main function. """
    ##################
    # Part 0: Setup. #
    ##################
    beta = 250.0
    num_outputs = 16

    ################################
    # Part 1: Define your ConvNet. #
    ################################
    # Create a new log directory
    # run: `tensorboard --logdir log` to see all the nice summaries
    for n_model in range(1000):
        LOG_DIR = 'log/model_%d' % n_model
        from os import path
        if not path.exists(LOG_DIR):
            break

    # Lets clear the tensorflow graph, so that you don't have to restart the
    # notebook every time you change the network
    tf.reset_default_graph()

    train_data = LoadDataset('final_run.tfrecords')
    valid_data = LoadDataset('final_run.tfrecords')

    # Create an iterator for the datasets.
    # The iterator allows us to quickly switch between training and validation.
    iterator = tf.contrib.data.Iterator.from_structure(
        train_data.output_types, ((None, 224, 224, 3), (None, 7)))

    # And fetch the next images from the dataset (every time next_image is
    # evaluated a new image set of 64 images is returned).
    next_image, next_label = iterator.get_next()

    # Define operations that switch between train and valid.
    switch_train_op = iterator.make_initializer(train_data)
    switch_valid_op = iterator.make_initializer(valid_data)

    # Convert the input.
    image = tf.cast(next_image, tf.float32)
    label = tf.cast(next_label, tf.float32)
    position_label = label[:, :3]
    rotation_label = label[:, 3:]

    # Whiten the input.
    # TODO(kirmani): Properly whiten the inputs.
    inputs = tf.identity(image, name='inputs')
    white_inputs = (inputs - 100.) / 72.

    with tf.name_scope('model'), tf.variable_scope('model'):
        h = white_inputs
        h = tf.contrib.layers.conv2d(
            h,
            num_outputs, [7, 7],
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None)
        h = tf.contrib.layers.batch_norm(h)
        h = tf.nn.relu(h)

        h = tf.contrib.layers.conv2d(
            h,
            num_outputs, [3, 3],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None)
        h = tf.contrib.layers.batch_norm(h)
        h = tf.nn.relu(h)

        for downsample in range(5):
            for resident_block in range(2):
                shortcut = h

                h = tf.contrib.layers.conv2d(
                    h,
                    num_outputs, [3, 3],
                    stride=1,
                    weights_regularizer=tf.nn.l2_loss,
                    activation_fn=None)
                h = tf.contrib.layers.batch_norm(h)
                h = tf.nn.relu(h)

                h = tf.contrib.layers.conv2d(
                    h,
                    num_outputs, [3, 3],
                    stride=1,
                    weights_regularizer=tf.nn.l2_loss,
                    activation_fn=None)
                h = tf.contrib.layers.batch_norm(h)
                h = h + shortcut
                h = tf.nn.relu(h)

            num_outputs *= 2

            h = tf.contrib.layers.conv2d(
                h,
                num_outputs, [3, 3],
                stride=2,
                weights_regularizer=tf.nn.l2_loss,
                activation_fn=None)
            h = tf.contrib.layers.batch_norm(h)
            h = tf.nn.relu(h)

        h = tf.contrib.layers.flatten(h)

        h = tf.contrib.layers.fully_connected(
            h,
            num_outputs=2048,
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None)
        h = tf.contrib.layers.batch_norm(h)
        h = tf.nn.relu(h)

        position = tf.contrib.layers.fully_connected(
            h,
            num_outputs=3,
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None)
        rotation = tf.contrib.layers.fully_connected(
            h,
            num_outputs=4,
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None)

        # Normalize the quaternion to unit length.
        rotation = rotation / tf.norm(rotation)

    # Define the loss function
    position_loss = tf.nn.l2_loss(position_label - position)
    rotation_loss = tf.nn.l2_loss(rotation_label - rotation)
    loss = position_loss + beta * rotation_loss

    # Let's weight the regularization loss down, otherwise it will hurt the
    # model performance.
    regularization_loss = tf.losses.get_regularization_loss()
    total_loss = loss + 1e-6 * regularization_loss

    # Metrics.
    position_error = tf.reduce_mean(tf.norm(position_label - position, axis=1))

    # TODO(kirmani): Figure out if this actually tells us the difference
    # between two quaternions.
    rotation_error = tf.reduce_mean(tf.norm(rotation_label - rotation, axis=1))

    # Adam will likely converge much faster than SGD for this assignment.
    optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)

    # use that optimizer on your loss function (control_dependencies makes sure any
    # batch_norm parameters are properly updated)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = optimizer.minimize(total_loss)

    # Let's define some summaries for tensorboard
    tf.summary.image('image', next_image, max_outputs=3)
    tf.summary.scalar('loss', tf.placeholder(tf.float32, name='loss'))
    tf.summary.scalar('position_loss',
                      tf.placeholder(tf.float32, name='position_loss'))
    tf.summary.scalar('rotation_loss',
                      tf.placeholder(tf.float32, name='rotation_loss'))
    tf.summary.scalar('position_error',
                      tf.placeholder(tf.float32, name='position_error'))
    tf.summary.scalar('rotation_error',
                      tf.placeholder(tf.float32, name='rotation_error'))

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

    # Let's compute the model size
    print("Total number of variables used ",
          np.sum(
              [v.get_shape().num_elements() for v in tf.trainable_variables()]))

    #####################
    # Part 2: Training. #
    #####################
    # Start a session
    sess = tf.Session()

    # Create a saver object for saving and loading variables.
    saver = tf.train.Saver(max_to_keep=2)

    # Set up training
    sess.run(tf.global_variables_initializer())

    # Run the training for some iterations
    for it in range(100):
        sess.run(switch_train_op)

        loss_vals = []
        position_loss_vals = []
        rotation_loss_vals = []
        position_error_vals = []
        rotation_error_vals = []
        # Run 10 training iterations and 1 validation iteration
        for i in range(10):
            (loss_val, position_loss_val, rotation_loss_val, position_error_val,
             rotation_error_val, _) = sess.run([
                 loss, position_loss, rotation_loss, position_error,
                 rotation_error, opt
             ])
            loss_vals.append(loss_val)
            position_loss_vals.append(position_loss_val)
            rotation_loss_vals.append(rotation_loss_val)
            position_error_vals.append(position_error_val)
            rotation_error_vals.append(rotation_error_val)

        # TODO(kirmani): Write validation tensorboard metrics.
        sess.run(switch_valid_op)

        # Let's update tensorboard
        summary_writer.add_summary(
            sess.run(
                merged_summary, {
                    'loss:0': np.mean(loss_vals),
                    'position_loss:0': np.mean(position_loss_vals),
                    'rotation_loss:0': np.mean(rotation_loss_vals),
                    'position_error:0': np.mean(position_error_vals),
                    'rotation_error:0': np.mean(rotation_error_vals),
                }), it)
        print('[%3d] Loss: %0.3f' % (it, np.mean(loss_vals)))

        saver.save(sess, LOG_DIR + '/model.ckpt', global_step=it)


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
