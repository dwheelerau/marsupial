#!/usr/bin/env python
# coding: utf-8

# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
export LD_LIBRARY_PATH=/media/dwheeler/spinner/Linux_space/miniconda3/envs/marsupial/lib:$LD_LIBRARY_PATH

import argparse

import os
import re
import json
import pathlib
import subprocess
import torch
#import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path 
from PIL import Image
from datetime import datetime

import torchvision

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-m", "--model", help="path to marsupial.ai model", 
                        default="weights/marsupial_72s.pt")

args = args_parser.parse_args()
print(args.model)

model = torch.hub.load('ultralytics/yolov5', 'custom', args.model)
'''

# In[ ]:





# Once the model is loaded, we can pass objects to it and save them in memory as predictions. 

# In[4]:


# Specify the path to an image file to predict on
#image = "data/*"
image = "data/koala1.jpeg"
#image = "data/koala2.jpg"
#image = "data/australian_magpie.jpg"

# Pass the image path to our model
results = model(image)

# If we print the result, we can see what we get and how long it took to process
print(results)

# OR we can print this as a nice dataframe:
results.pandas()
results.xyxy[0]  # im predictions (tensor)
results.pandas().xyxy[0]  # im predictions (pandas)


# Our model has found one koala in the image; let's visualise the model's predictions to see if they look reasoable:

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(results.render()[0])
plt.show()


# That looks pretty good! The model has drawn a bounding box perfectly covering the one animal in this image, which looks to pretty clearly be a Koala walking along the ground.
# 
# #### Building functions with marsupial's models
# 
# Now that we can make a prediction on one single image, we can then build some functions to allow us to make predictions on multiple images.
# We can also make functions to help us visualise and summarise results.

# In[6]:


"""
Prediction functions
"""

def predict_single_image(image_file):
    
    im = image_file
    results = model(im)
    results.pandas().xyxy[0] 
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    plt.show()
    
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


# We can then pass a path to a directory containing images, and this function will recursively look for any images, load them in, and make predictions on all of the images using our model.
# 
# Predictions will then be saved into a `pandas` dataframe, which can be analysed in Python or easily exported into a spreadsheet for analysis using other tools like R or Excel,

# In[80]:


#predictions = predict_images("data") 
#predictions


# We can also print out a summary of the number of observations for each species detected:

# In[8]:


# We can enable printing a summary by setting summary to True
predictions = predict_images("data", summary = True) 

# OR we can always just do it ourselves with the output of predict_images:
# print(predictions.groupby(['name'])['name'].count().sort_values(ascending=False))


# In[7]:


predictions = predict_images("data/") 
 
predictions


# ### Gradio web applications
# 
# Here's an example of simple web application that can use one of our models to make predictions on input images:

# In[85]:


# Marsupial Demo
import gradio as gr
import torch
import torchvision
import numpy as np
from PIL import Image

# Load Marsupial model
# TODO: Allow user selectable model?

model = torch.hub.load('ultralytics/yolov5:v6.2', 'custom', "weights/marsupial_72s_lures.pt", trust_repo=True)

def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size))  # resize
    
    #model = torch.hub.load('ultralytics/yolov5', 'custom', "weights/marsupial_72s_lures.pt")
    
    results = model(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    result = Image.fromarray(results.imgs[0])
    return result

inputs = gr.Image(type="pil", label="Input Image")
outputs = gr.Image(type="pil", label="Output Image")

title = "Marsupial"
description = "Detect and identify 72 different species of Australian wildlife using Marsupial's most detailed model"
article = "<p style='text-align: center'>This app makes predictions using a YOLOv5s model that was trained to detect and identify 72 different species of animals found in Australia in camera trap images; find out more about the project on <a href='https://github.com/Sydney-Informatics-Hub/marsupial'>GitHub</a>. This app was built by Dr Henry Lydecker and Dr Gordon MacDonald at the Sydney Informatics Hub, a Core Research Facility at the University of Sydney. Find out more about the YOLO model from the original creator, <a href='https://pjreddie.com/darknet/yolo/'>Joseph Redmon</a>. YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset and developed by Ultralytics, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"

examples = [['data/eastern_grey_kangaroo.jpg'],['data/red_fox.jpg'], ['data/koala2.jpg'],['data/cat1.jpg']]

gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, theme="default").launch(enable_queue=True)


# In[ ]:



'''
