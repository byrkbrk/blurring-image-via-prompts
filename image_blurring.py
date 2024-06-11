import os
import torch
from transformers import pipeline
from ultralytics import SAM
from PIL import Image, ImageFilter



class BlurImage(object):
    def __init__(self):
        pass