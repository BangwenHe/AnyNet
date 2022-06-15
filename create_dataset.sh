#!/usr/bin/env bash

scenflow_data_path=/root/data/Sampler

monkaa_frames_cleanpass=$scenflow_data_path"/Monkaa/RGB_cleanpass"
monkaa_disparity=$scenflow_data_path"/Monkaa/disparity"
driving_frames_cleanpass=$scenflow_data_path"/Driving/RGB_cleanpass"
driving_disparity=$scenflow_data_path"/Driving/disparity"
flyingthings3d_frames_cleanpass=$scenflow_data_path"/FlyingThings3D/RGB_cleanpass"
flyingthings3d_disparity=$scenflow_data_path"/FlyingThings3D/disparity"

if [[ ! -d dataset ]] 
then
    mkdir dataset
fi

ln -s $monkaa_frames_cleanpass dataset/monkaa_frames_cleanpass
ln -s $monkaa_disparity dataset/monkaa_disparity
ln -s $flyingthings3d_frames_cleanpass dataset/frames_cleanpass
ln -s $flyingthings3d_disparity dataset/frames_disparity
ln -s $driving_frames_cleanpass dataset/driving_frames_cleanpass
ln -s $driving_disparity dataset/driving_disparity

