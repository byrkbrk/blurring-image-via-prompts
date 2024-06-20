import os
import torch
from transformers import pipeline
from ultralytics import SAM
from PIL import Image, ImageFilter
from torchvision import transforms



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
        self.create_dirs(self.module_dir)

    def blur(self, image, text_prompts, labels=None, save=True, size=None):
        """Returns blurred image based on given text prompt"""
        if type(image) == str: 
            image_name = image
            image = self.read_image(self.module_dir, "images-to-blur", image, size)
        else:
            image_name = "noname"
            if size: image = self.resize_image(image, size)

        aggregated_mask = self.get_aggregated_mask(image, text_prompts, labels)
        blurred_image = self.blur_entire_image(image)
        blurred_image[:, ~aggregated_mask] = transforms.functional.pil_to_tensor(image)[:, ~aggregated_mask]
        blurred_image = transforms.functional.to_pil_image(blurred_image)
        if save:
            blurred_image.save(os.path.join(self.module_dir, "blurred-images", os.path.splitext(image_name)[0] + "_blurred_image.jpg"))
        return blurred_image

    def get_aggregated_mask(self, image, text_prompts, labels=None):
        """Returns aggregated mask for given image, text prompts, and labels"""
        aggregated_mask = 0
        for annotation in self.get_annotations(image, text_prompts, labels):
            aggregated_mask += annotation["segmentation"].float()
        return aggregated_mask > 0
    
    def get_annotations(self, image, text_prompts, labels=None):
        """Returns annotations predicted by SAM"""
        return self.annotate_sam_inference(*self.predict_masks(image, 
                                                               self.predict_boxes(image, text_prompts),
                                                               labels))

    def predict_masks(self, image, box_prompts, labels=None):
        """Returns predicted masks by SAM"""
        if labels is None:
            labels = [1]*len(box_prompts)
        return self.sam(image, bboxes=box_prompts, labels=labels)
    
    def predict_boxes(self, image, prompts):
        """Returns bounding boxes for given image and prompts"""
        return [list(pred["box"].values()) for pred in self.owlvit(image, prompts)]

    def initialize_device(self, device):
        """Initializes device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    def blur_entire_image(self, image, radius=50):
        """Returns Gaussian-blurred image"""
        return transforms.functional.pil_to_tensor(image.filter(ImageFilter.GaussianBlur(radius=radius)))
    
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
            annotation["area"] = annotation["segmentation"].sum()

            if annotation["area"] >= area_threshold:
                annotations += [annotation]
        return annotations
    
    def read_image(self, root, base_folder, image_name, size=None):
        """Returns the openned for given image name base folder, and root"""
        image = Image.open(os.path.join(root, base_folder, image_name))
        if size:
            image = self.resize_image(image, size)
        return image
    
    def resize_image(self, image, size):
        """Returns resized image"""
        return image.resize(size)
    
    def create_dirs(self, root):
        """Creates required directories under the given root"""
        dir_names = ["images-to-blur", "blurred-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)
        

if __name__ == "__main__":
    image = Image.open("images-to-blur/dogs.jpg")
    image = image.resize((1024, 1024))
    blur_image = BlurImage(device="cpu")
    #annotations = blur_image.get_annotations(image, ["small nose"])
    #print(annotations)
    #print(blur_image.get_aggregated_mask(image, ["small nose"]).shape)
    image = "dogs.jpg"
    blur_image.blur(image, ["jacket"])