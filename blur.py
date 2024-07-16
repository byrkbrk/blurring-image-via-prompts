from argparse import ArgumentParser
from image_blurring import BlurImage



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Blurs image based on given text prompts")
    parser.add_argument("image_name", type=str, default=None, 
                        help="Name of the image file that be processed. Image file must be in `images-to-blur` folder")
    parser.add_argument("text_prompts", nargs="+", type=str, default=None, 
                        help="Text prompts for the objects that get blurred")
    parser.add_argument("--blur_intensity", type=int, default=50, 
                        help="Intensity of the blur that be applied. Default: 50")
    parser.add_argument("--image_size", nargs="+", type=int, default=None,
                        help="Size (width, height) to which the image be transformed. Default: None")
    parser.add_argument("--device", type=str, default=None, help="Device that be used during inference. Default: None")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    BlurImage(device=args.device).blur(args.image_name, args.text_prompts, args.blur_intensity, size=args.image_size)
