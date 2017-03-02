#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_cf_imdb, get_imdb, list_imdbs
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb
import os
import subprocess
    
def sync_location(location, local_dir="."):
    """
    Copies folders and or files from google cloud storage to local storage
    """
    if location is None:
        return None
    result = location.split("/")[-1]
    if location.startswith("gs://"):  
        if local_dir is not ".":
            result = local_dir
            #Cache
            if os.path.exists(result):
                return result
            else:
                subprocess.check_call(['mkdir', '-p', local_dir])
        # Cache
        if os.path.exists(result):
            return result
        subprocess.check_call(['gsutil', '-qm', 'cp', '-r', location, local_dir])
        
    return result


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default=None, type=str)
    parser.add_argument('--imdb_data_url', dest='imdb_data_url',
                        help='A location to download the dataset from',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='image_path',
                        help='path to images for training',
                        default=None, type=str)
    parser.add_argument('--label_path', dest='label_path',
                        help='path to label data for training',
                        default=None, type=str)
    parser.add_argument('--label_type', dest='label_type',
                        help='type of labeled data (json or csv)',
                        default=None, type=str)
    parser.add_argument('--class_names_path', dest='class_names_path',
                        help='path to class names for training',
                        default=None, type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='path to store your output',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    label_path = sync_location(args.label_path, "tmp_labels")
    image_path = sync_location(args.image_path, "tmp_images")
    class_names_path = sync_location(args.class_names_path)
    cfg_file = sync_location(args.cfg_file)
    cfg_from_file(cfg_file)

    pretrained_model = None
    if args.pretrained_model is not None:
        pretrained_model = sync_location(args.pretrained_model)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        
    print list_imdbs()
    if args.imdb_name is None:
        print 'Loaded CrowdFlower dataset for training'
        imdb = get_cf_imdb(label_path, image_path, class_names_path, args.label_type)
    else:
        print 'Loading IMDB %s' % args.imdb_name
        # TODO: this is hardcoded to VOC for now...
        if args.imdb_data_url is not None:
            sync_location(args.imdb_data_url, os.path.join(cfg.DATA_DIR, 'VOCdevkit2007'))
        imbd = get_imdb(args.imdb_name)
        print imdb

    roidb = get_training_roidb(imdb)

    output_dir = args.output_path
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/{}:{:d}'.format(args.device,args.gpu_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=args.max_iters)
