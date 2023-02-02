#!/usr/bin/env python
# coding: utf-8

# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
#export LD_LIBRARY_PATH=/media/dwheeler/spinner/Linux_space/miniconda3/envs/marsupial/lib:$LD_LIBRARY_PATH

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
args_parser.add_argument("-i", "--image_dir", help="folder of images to analyse", 
                        required=True)
args_parser.add_argument("-o", "--out_dir", help="folder of images to analyse", 
                        default="./")

args = args_parser.parse_args()
#print(args.model)

model = torch.hub.load('ultralytics/yolov5', 'custom', args.model)


#image = "data/koala1.jpeg"
#results = model(image)
#print(results)

# OR we can print this as a nice dataframe:
#results.pandas()
#results.xyxy[0]  # im predictions (tensor)
#results.pandas().xyxy[0]  # im predictions (pandas)

#get_ipython().run_line_magic('matplotlib', 'inline')
#fig, ax = plt.subplots(figsize=(16, 12))
#ax.imshow(results.render()[0])
#plt.show()

# #### Building functions with marsupial's models

"""
Prediction functions
"""

def predict_single_image(image_file):
    
    im = image_file
    results = model(im)
    results.pandas().xyxy[0] 
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    outfile="./"+im
    plt.save(outfile)
    
    return(results.pandas().xyxy[0])

def predict_images(image_dir, summary = False):
    
    predictions = [] 
    
    image_dir = pathlib.Path(image_dir)
    
    for image in tqdm(image_dir.glob("*.jpg")):
        
        prediction = predict_single_image(image)
        prediction['fullpath'] = image
        prediction['filename'] = os.path.basename(image) 
        predictions.append(prediction)
        
                       
    prediction_df = pd.concat(predictions)
    
    if summary == True:
        print("Detection Summary")
        print(prediction_df.groupby(['name'])['name'].count().sort_values(ascending=False))
        
    return(prediction_df)

"""
Image EXIF utilities

This set of functions is to extract information from image EXIF data. 

These functions are very useful when working with EXIF anotated images, like the NSWNP Wildcount dataset.
"""

def species_annotation_from_exif(image_file):
    # Get original species ID
    cmd="exiftool -Keywords " + '''"''' + str(image_file) + '''"'''
    image_keywords = subprocess.getoutput([cmd])
    species = re.findall(r'(?<=a\)\s).+?(?=\s\([a-zA-z])', image_keywords)
    return(species)

def date_time_original_from_exif(image_file):
    # Get DateTimeOriginal
    cmd="exiftool -DateTimeOriginal " + '''"''' + str(image_file) + '''"'''
    date_time_original = subprocess.getoutput([cmd])
    date_time_original = re.findall(r'(?<=:\s)(.*)', date_time_original)
    return(date_time_original)

def camera_name_from_exif(image_file):
    # Get Camera Name
    cmd="exiftool -UserLabel " + '''"''' + str(image_file) + '''"'''
    camera_name = subprocess.getoutput([cmd])
    camera_name = re.findall(r'(?<=:\s)(.*)', camera_name)
    return(camera_name)

def date_from_exif(image_file):
    # Get date
    cmd = "exiftool -DateTimeOriginal " + '''"''' + str(image_file) + '''"'''
    date = subprocess.getoutput([cmd])
    date = re.findall(r'(?<=:\s)([^\s]+)', date)
    return(date)

def time_from_exif(image_file):
    # Get time
    cmd = "exiftool -DateTimeOriginal " + '''"''' + str(image_file) + '''"'''
    time = subprocess.getoutput([cmd])
    time = time[time.rindex(' ')+1:]
    return(time)



predictions = predict_images(image_dir=args.image_dir) 
#predictions

# We can enable printing a summary by setting summary to True
#predictions = predict_images("data", summary = True) 

#predictions = predict_images("data/") 
 
#predictions
