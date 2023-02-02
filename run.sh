#!/usr/bin/bash

# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
export LD_LIBRARY_PATH=/media/dwheeler/spinner/Linux_space/miniconda3/envs/marsupial/lib:$LD_LIBRARY_PATH

# defaults
out_dir="processed_images"
model="weights/marsupial_72s.pt"

Help()
{
   # Display Help
   echo "Run script for prediction_batch.py."
   echo "Syntax: scriptTemplate [-m|a|h]"
   echo "options:"
   echo "m     Path to model .pb file"
   echo "i     Path to directory containing target images"
   echo "o     Out directory of processed images with Bbox."
   echo "h     Print this Help."
}

while getopts ":i:m:o:h" opt; do
    case ${opt} in
        i ) image_dir=${OPTARG};;
        o ) out_dir=${OPTARG};;
        m ) model=${OPTARG};;
        h ) Help
            exit;;
    esac
done

echo $model
echo $image_dir
echo $out_dir

python prediction_batch.py -m $model -i $image_dir -o $out_dir

# unset or will cause stability issues
unset LD_LIBRARY_PATH
