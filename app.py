import spaces
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from ormbg.models.ormbg import ORMBG
from PIL import Image

model_path = "models/ormbg.pth"

# Load the model globally but don't send to device yet
net = ORMBG()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()


def resize_image(image):
    image = image.convert("RGB")
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


@spaces.GPU
@torch.inference_mode()
def inference(image):
    # Check for CUDA and set the device inside inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Prepare input
    orig_image = Image.fromarray(image)
    w, h = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = torch.divide(im_tensor, 255.0)

    if torch.cuda.is_available():
        im_tensor = im_tensor.to(device)

    # Inference
    result = net(im_tensor)
    # Post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    # Image to PIL
    im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    # Paste the mask on the original image
    new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)

    return new_im


# Gradio interface setup
title = "Open Remove Background Model (ormbg)"
description = r"""
This model is a <strong>fully open-source background remover</strong> optimized for images with humans. It is based on [Highly Accurate Dichotomous Image Segmentation research](https://github.com/xuebinqin/DIS). The model was trained with the synthetic <a href="https://huggingface.co/datasets/schirrmacher/humans">Human Segmentation Dataset</a>, <a href="https://paperswithcode.com/dataset/p3m-10k">P3M-10k</a> and <a href="https://paperswithcode.com/dataset/aim-500">AIM-500</a>.

If you identify cases where the model fails, <a href='https://huggingface.co/schirrmacher/ormbg/discussions' target='_blank'>upload your examples</a>!

- <a href='https://huggingface.co/schirrmacher/ormbg' target='_blank'>Model card</a>: find inference code, training information, tutorials
- <a href='https://huggingface.co/schirrmacher/ormbg' target='_blank'>Dataset</a>: see training images, segmentation data, backgrounds
- <a href='https://huggingface.co/schirrmacher/ormbg\#research' target='_blank'>Research</a>: see current approach for improvements
"""

examples = [
    "./examples/image/example1.jpeg",
    "./examples/image/example2.jpeg",
    "./examples/image/example3.jpeg",
]

demo = gr.Interface(
    fn=inference,
    inputs="image",
    outputs="image",
    examples=examples,
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=["./"])
