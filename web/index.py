# import gradio as gr
# from transformers import *
# gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()

# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# demo.launch()


# import gradio as gr
# import torch
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import os



# def process_image(image_path):
#     image_path = Path(image_path)
#     image_raw = Image.open(image_path)
#     image = image_raw.resize(
#         (800, int(800 * image_raw.size[1] / image_raw.size[0])),
#         Image.Resampling.LANCZOS)

#     # prepare image for the model
#     encoding = feature_extractor(image, return_tensors="pt")

#     # forward pass
#     with torch.no_grad():
#         outputs = model(**encoding)
#         predicted_depth = outputs.predicted_depth


#     return [img, gltf_path, gltf_path]

# title = "Demo: zero-shot depth estimation with DPT + 3D Point Cloud"
# description = "This demo is a variation from the original DPT Demo. It uses the DPT model to predict the depth of an image and then uses 3D Point Cloud to create a 3D object."
# examples = [["examples/1-jonathan-borba-CgWTqYxHEkg-unsplash.jpg"]]

# iface = gr.Interface(fn=process_image,
#                      inputs=[gr.Image(
#                          type="filepath", label="Input Image")],
#                      outputs=[gr.Image(label="predicted depth", type="pil"),
#                               gr.Model3D(label="3d mesh reconstruction", clear_color=[
#                                                  1.0, 1.0, 1.0, 1.0]),
#                               gr.File(label="3d gLTF")],
#                      title=title,
#                      description=description,
#                      examples=examples,
#                      allow_flagging="never",
#                      cache_examples=False)

# iface.launch(debug=True, enable_queue=False)

# import gradio as gr
# def segment(image):
#     pass  # Implement your image segmentation model here...

# gr.Interface(fn=segment, inputs="image", outputs="image").launch()


import gradio as gr
from torchvision import transforms


def predict(inp):

    return "test"

# demo = gr.Interface(fn=predict, 
#               inputs="image", outputs="text"
#              )
demo = gr.Interface(fn=predict, 
              inputs="text", outputs="image"
             )
             
demo.launch()