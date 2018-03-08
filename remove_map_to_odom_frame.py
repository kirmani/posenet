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
import sys
import traceback
import time
import rosbag


def main(args):
    """ Main function. """
    filename = "final_run"
    bag = rosbag.Bag(filename + '.bag')
    filtered = rosbag.Bag(filename + '-filtered.bag', 'w')
    for topic, msg, t in bag.read_messages():
        if topic == '/tf':
            if (msg.transforms[0].header.frame_id == 'map' and
                    msg.transforms[0].child_frame_id == 'odom'):
                print(msg)
                # msg.transforms[0].child_frame_id = 'old_odom'
            else:
                filtered.write(topic, msg, t)
        else:
            filtered.write(topic, msg, t)
    bag.close()
    filtered.close()


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
