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
        """Returns bounding boxes for given image and prompts"""
        return [list(pred["box"].values()) for pred in self.owlvit(image, prompts)]

    def initialize_device(self, device):
        pass

    def blur_entire_image(self, image, radius=50):
        """Returns Gaussian-blurred image"""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def annotate_sam_inference(self, inference, area_threshold=0):
        """Returns list of annotation dicts
        Args:
            inference (ultralytics.engine.results.Results): Output of the model
            area_threshold (int): Threshold for the segmentation area
        """
        annotations = []
        for i in range(len(inference.masks.data)):
            annotation = {}
            annotation["id"] = i
            annotation["segmentation"] = inference.masks.data[i].cpu()==1
            annotation["bbox"] = inference.boxes.data[i]
            annotation["score"] = inference.boxes.conf[i]
            annotation["area"] = annotation["segmentation"].sum()

            if annotation["area"] >= area_threshold:
                annotations += [annotation]
        return annotations


if __name__ == "__main__":
    blur_image = BlurImage()