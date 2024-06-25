import gradio as gr
from image_blurring import BlurImage



if __name__ == "__main__":
    blur_image = BlurImage()
    gr_interface = gr.Interface(
        fn=lambda image, prompt, save=False: blur_image.blur(image, prompt.split("\n"), save=save),
        inputs=[gr.Image(type="pil"), gr.Textbox(lines=3, placeholder="jacket\ndog head\netc...")],
        outputs=gr.Image(type="pil")
    )
    gr_interface.launch()
