import os
import gradio as gr
from image_blurring import BlurImage



if __name__ == "__main__":
    blur_image = BlurImage(device=None)
    gr_interface = gr.Interface(
        fn=lambda image, prompt, intensity, save=False: blur_image.blur(image, prompt.split("\n"), intensity, save=save),
        inputs=[gr.Image(type="pil"), 
                gr.Textbox(lines=3, placeholder="jacket\ndog head\netc..."),
                gr.Slider(minimum=0, maximum=400, step=10, value=50, label="Blur intensity")],
        outputs=gr.Image(type="pil"),
        title="Blur Objects by Prompts",
        examples=[
            [os.path.join(os.path.dirname(__file__), 
                          "images-to-blur", 
                          "dogs.jpg"),
             "jacket\ndog head"],
             [os.path.join(os.path.dirname(__file__), 
                          "images-to-blur", 
                          "hat_sunglasses.jpg"),
              "hat\nsunglasses"]             
        ]
    )
    gr_interface.launch()
