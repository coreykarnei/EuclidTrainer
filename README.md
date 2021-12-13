# EuclidTrainer
Pose Estimation + Weightlifting

## Setup

Setup your conda environment so that you have all the packages in the `requirements.yml` file. Then download the alphapose model file [here](https://drive.google.com/file/d/1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn/view) or visit their [model zoo](https://drive.google.com/file/d/1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn/view). Put the downloaded file into a folder called "model".

## Usage

To run the system, call:
```
python main.py --video $VIDEO_PATH --exercise $EXERCISE_TYPE
```
Exercise type can be either "squat" or "clean.