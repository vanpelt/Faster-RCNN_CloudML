# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Uber ATG
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from fast_rcnn.config import cfg
import pdb
import glob
import json

class crowdflower(imdb):
    """
    Loads images and annotation generated by the CrowdFlower platform into the imdb format
    """
    def __init__(self, label_path, image_path, class_names_path, label_type):
        imdb.__init__(self, 'cfimdb')

        self._classes = self._load_classes_from_path(class_names_path)
        
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        
        self._label_path = label_path
        self._label_type = label_type
        self._image_path = image_path

        self._image_index = glob.glob(os.path.join(self._image_path, "*%s" % self._image_ext))

        if len(self.image_index) == 0:
            raise ValueError("Didn't find any images in your path %s with extension %s" % (self._image_path, self._image_ext))

        
        newList = []
        for i in range(len(self._image_index)):
          if self._label_type=="csv":
              jpgName = self._image_index[i]
              labelBase = os.path.basename(jpgName[:-4]+".csv")
              fname = os.path.join(self._label_path, labelBase)

              if os.path.isfile(fname) and os.stat(fname).st_size > 0:
                  newList.append(self._image_index[i])
          elif label_type=="json":
              jpgName = self._image_index[i]
              jsonBase = os.path.basename(jpgName[:-4]+".json")
              fname = os.path.join(self._label_path, jsonBase)

              if os.path.isfile(fname) and self._check_json_has_box(fname):
                  newList.append(self._image_index[i])
          else:
              raise ValueError("Label Type must be csv or json, was: %s" % label_type)
          
        self._image_index = sorted(newList)

        if len(self._image_index)==0:
            raise ValueError("Didn't find any matchined data files in your path %s" % self._label_path)

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

    def _load_classes_from_path(self, class_names_path):
        with open(class_names_path) as json_file:
            data = json.load(json_file)
        return data
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        return self._image_index[index]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if False and os.path.exists(cache_file): # scaffold - disable cache
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._label_type == 'csv':
            gt_roidb = [self._load_csv_annotation(index) for index in range(len(self.image_index))]
        elif self._label_type == 'json':
            gt_roidb = [self._load_json_annotation(index) for index in range(len(self.image_index))]
        else:
            raise ValueError("Label Type must be csv or json, was: %s", self._label_type)
        
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
      assert(False)
      pass

    def rpn_roidb(self):
      assert(False)
      pass

    def _load_rpn_roidb(self, gt_roidb):
      assert(False)
      pass

    def _load_selective_search_roidb(self, gt_roidb):
      assert(False)
      pass

   # CrowdFlower Data Format:
   # {"shapes":[{"probability":0.9594706892967224,"height":58,"width":19,"y":597,"x":1350,
   # "type":"predicted_box","id":0,"hidden":false,"active":false},{"probability":0.9342359304428101,
   # "height":47,"width":32,"y":608,"x":1490,"type":"predicted_box","id":1,"hidden":false,"
   # active":true},{"type":"box","x":1365,"y :599,"width":21,"height":58,"id":1484099731,"active":false}]}

    def _check_json_has_box(self, fname):
        if (os.stat(fname).st_size == 0):
            return False
        
        with open(fname) as data_file:
            data = json.load(data_file)

        
            
        row_data = data['shapes']
            
        box_row_data = [d for d in row_data if (d["type"] == "box" or d["type"] == "predicted_box")]
        num_objs = len(box_row_data)
        return num_objs > 0
            
    def _load_json_annotation(self, index):

        width = self._get_width(index)
        height = self._get_height(index)

        jsonBase = os.path.basename(self._image_index[index])[:-4]+".json"
        fname = os.path.join(self._label_path, jsonBase)
        
        with open(fname) as data_file:
            data = json.load(data_file)

        row_data = data['shapes']
        #print row_data
        box_row_data = [d for d in row_data if (d["type"] == "box" or d["type"] == "predicted_box")]
        num_objs = len(box_row_data)
        
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, box_data in enumerate(box_row_data):
            raw_x1 = float(box_data['x'])
            raw_x2 = raw_x1 + float(box_data['width'])
            raw_y1 = float(box_data['y'])
            raw_y2 = raw_y1 + float(box_data['height'])

            raw_x1 = max(min(raw_x1, width-1), 0)
            raw_x2 = max(min(raw_x2, width-1), 0)
            raw_y1 = max(min(raw_y1, height-1), 0)
            raw_y2 = max(min(raw_y2, height-1), 0)

            x1 = min(raw_x1, raw_x2)
            x2 = max(raw_x1, raw_x2)
            y1 = min(raw_y1, raw_y2)
            y2 = max(raw_y1, raw_y2)
            
            
            if not (x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0):
                raise ValueError()
            
            if not(x2 >= x1 and y2 >= y1):
                raise ValueError()
            
            cls = 1 # not handling multuple classes
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        #print "Output: %s" % boxes
        
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}    

    def _load_csv_annotation_file(self, index):
        csvBase = os.path.basename(self._image_index[index])[:-4]+".csv"
        fname = os.path.join(self._label_path, csvBase)
        gts = np.genfromtxt(fname, delimiter=' ')
        if len(gts.shape) == 1:
          gts = gts.reshape((1, len(gts)))
        num_objs = len(gts)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(gts):
            # Looks like pixel indexes in gt csv files are 1-based
            x1, y1, x2, y2 = float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
            assert(x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0)
            assert(x2 >= x1 and y2 >= y1)
            cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
      pass

    def _get_voc_results_file_template(self):
      pass

    def _write_voc_results_file(self, all_boxes):
      pass

    def _do_python_eval(self, output_dir = 'output'):
      pass

    def _do_matlab_eval(self, output_dir='output'):
      pass

    def evaluate_detections(self, dets, output_dir, thresh=0.8):
      gts = self.gt_roidb()
      for iImage in range(len(dets[0])):
        for iClass in range(1, self.num_classes):
          preds = dets[iClass][iImage]
          clipPreds = np.minimum(30, len(preds))
          bboxes = preds[:clipPreds, :4]
          scores = preds[:clipPreds, -1]
          bboxesConf = np.compress(scores > thresh, bboxes, axis=0)
          if len(bboxesConf.shape) == 1:
            bboxesConf = bboxesConf.reshape((1,-1))
          # Get IoU overlap between each ex ROI and gt ROI
          gtBoxes = gts[iImage]['boxes'].astype(np.float)
          """
          Parameters for bbox_overlaps
          ----------
          boxes: (N, 4) ndarray of float
          query_boxes: (K, 4) ndarray of float
          Returns
          -------
          overlaps: (N, K) ndarray of overlap between boxes and query_boxes
          """
          det_gt_overlaps = utils.cython_bbox.bbox_overlaps(bboxesConf.astype(np.float), gtBoxes)

          FN = 0
          TP = 0
          IOU = 0.2 # TODO: Parametrize
          # dets = rows, gts = cols
          # for a given det, is there an overlapping gt?
          for row in det_gt_overlaps:
            if len(row) > 0 and row.max() >= IOU:
              TP = TP+1
          # for a given gt, is there no overlapping det?
          for row in det_gt_overlaps.transpose():
            if len(row) > 0 and row.max() < IOU:
              FN = FN+1
          # example: [[ 0.          0.78478233  0.          0.          0.          0.          0.        ]]
          # MR = FN/(FN+TP)
          MR = 0 if (FN+TP == 0) else (FN/float(FN+TP))
          print("MR: %.3f" % MR)
          print("FN %d TP %d" % (FN, TP))

    def competition_mode(self, on):
      pass
