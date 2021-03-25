# Face Mask Detector

Code adapted from [JadHADDAD92/covid-mask-detector](https://github.com/JadHADDAD92/covid-mask-detector)

## Train your own network

```Shell
cd mask_detector

# get the RFMD dataset
python data_utils/get_process_RFMD.py

# train the network
python train.py
```

## Run the mask detector on a YT video

```Shell
cd mask_detector

# get a sample youtube video
python data_utils/get_YT_video.py

# run the detector
python run.py
```

![gif](vid.gif)
