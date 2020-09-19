import os
import xml.etree.ElementTree as ET
import cv2
from detectron2.structures import BoxMode
from utils.parameters import *

def get_hrb_dicts(image_set):
    
    img_dir = HRBParis_imgDir
    image_set_file =  HRBParis_Dir + image_set + '.txt'
    image_ext = '.jpg'
    annotations_dir = HRBParis_annoDir

    classes = ORGAN_CLASSES

    class_to_ind = dict(zip(classes, range(len(classes))))

    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    dataset_dicts = []
    for idx in image_index:
        record = {}
        
        filename = os.path.join(img_dir, idx + image_ext)
        height, width = cv2.imread(filename).shape[:2]

        annotation_file = annotations_dir + idx + '.xml'
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        tree = ET.parse(annotation_file)
        objects = tree.findall('object')
        
        #annos = v["regions"]
        objs = []
        for label in objects:
            #assert not anno["region_attributes"]
            #anno = anno["shape_attributes"]
            #px = anno["all_points_x"]
            #py = anno["all_points_y"]
            #poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #poly = [p for x in poly for p in x]

            bbox = label.find('bndbox')

            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cat = label.find('name').text.lower().strip()
            class_indx = class_to_ind[cat]          


            obj = {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_indx
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_frhrb_dicts(d):
    
    img_dir = HRBFR_imgDir
    image_set_file = HRBFR_Dir + 'list.txt'
    image_ext = '.jpg'
    annotations_dir = HRBFR_annoDir

    classes = ORGAN_CLASSES

    class_to_ind = dict(zip(classes, range(len(classes))))

    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    dataset_dicts = []
    index_len = len(image_index)
    counter = 1
    
    for idx in image_index:
        progress = (counter / index_len) * 100
        print(' Progress:', f'{progress:.5f}%', end='\r')
        record = {}
        
        filename = os.path.join(img_dir, idx + image_ext)
        height, width = cv2.imread(filename).shape[:2]

        annotation_file = annotations_dir + idx + '.xml'
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        tree = ET.parse(annotation_file)
        objects = tree.findall('object')
      
        #annos = v["regions"]
        objs = []
        counter += 1
        for label in objects:
            #assert not anno["region_attributes"]
            #anno = anno["shape_attributes"]
            #px = anno["all_points_x"]
            #py = anno["all_points_y"]
            #poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #poly = [p for x in poly for p in x]

            bbox = label.find('bndbox')

            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cat = label.find('name').text.lower().strip()
            class_indx = class_to_ind[cat]            


            obj = {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                #"segmentation": [poly],
                "category_id": class_indx,
                #"iscrowd": 0
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
