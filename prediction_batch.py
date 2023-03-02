#!/usr/bin/env python
# coding: utf-8

import argparse

import os

import re
import json
import pathlib
import subprocess
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path 
from PIL import Image
from datetime import datetime

import torchvision
import matplotlib.pyplot as plt

args_parser = argparse.ArgumentParser()

args_parser.add_argument("-m", "--model", help="path to marsupial.ai model",
                        default="weights/marsupial_72s.pt")
args_parser.add_argument("-o", "--out_dir", help="output directory for bbox images", 
                        default="processed_images")
args_parser.add_argument("-i", "--image_dir", help="folder of images to analyse", 
                        required=True)

args = args_parser.parse_args()
print("model being used is:")
print(args.model)

# create the outdir if it does not exist
try:
    os.mkdir(args.out_dir)
except FileExistsError:
    pass
#https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'custom', args.model)

def predict_images(image_dir, out_dir, summary = True):
    
    predictions = [] 
    
    image_dir = pathlib.Path(image_dir)

    target_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir)
                     for f in filenames 
                     if os.path.splitext(f.lower())[1] == '.jpg']
    
    for im in tqdm(target_images): #tqdm(image_dir.glob("*.jpg")):
        try:
            results = model(im)
        except OSError:
            print("Image file %s is broken, skipping!" % im)
        results.pandas().xyxy[0] 
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(results.render()[0])
        
        # remove leading slash from im dirname to get join to work
        out_dir = Path(out_dir)
        image_dir_name = Path(os.path.dirname(im))
        
        bbox_dir = out_dir / image_dir_name
        bbox_file = os.path.basename(im).lower().replace(".jpg",
                                                         "_detections.jpg").replace(".JPG","_detections.jpg")
        try:
            os.makedirs(bbox_dir)
        except FileExistsError:
            pass

        outfile=os.path.join(bbox_dir, bbox_file)
        plt.savefig(outfile)
    
        prediction = results.pandas().xyxy[0]
        prediction['fullpath'] = im
        prediction['filename'] = os.path.basename(im) 
        predictions.append(prediction)
        
                       
    prediction_df = pd.concat(predictions)
    prediction_df.to_csv("predictions.csv", index=False) 

    if summary == True:
        print("Detection Summary")
        print(prediction_df.groupby(['name'])['name'].count().sort_values(ascending=False))
        
    return(prediction_df)

# make prediction on a folder of images
predictions = predict_images(image_dir=args.image_dir, out_dir=args.out_dir) 
