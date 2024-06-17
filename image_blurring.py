import os
import torch
from transformers import pipeline
from ultralytics import SAM
from PIL import Image, ImageFilter



class BlurImage(object):
    def __init__(self, 
                 device="mps",
                 owlvit_ckpt="google/owlv2-base-patch16-ensemble",
                 owlvit_task="zero-shot-object-detection",
                 sam_ckpt="mobile_sam.pt",
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.owlvit = pipeline(model=owlvit_ckpt, task=owlvit_task, device=self.device)
        self.sam = SAM(model=sam_ckpt)

    def predict_masks(self, image, prompts):
        pass
    
    def predict_boxes(self, image, prompts):
        pass

    def initialize_device(self, device):
        pass


if __name__ == "__main__":
    blur_image = BlurImage()