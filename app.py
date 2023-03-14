import torch
from torch import autocast
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import time

import base64
from io import BytesIO
import os
from PIL import Image

def init():
    # Midas
    global midas_model
    global transform

    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas_model.to("cuda")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    # Controlnet
    global pipe
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()



def inference(model_inputs:dict, img_bytes, debug=False):
    global midas_model
    global transform
    global pipe

    input_img = Image.open(BytesIO(img_bytes))
    image = input_img.convert('RGB')

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    if not image: return {'message': 'No image was provided'}
    
    # Run Midas
    image_array = np.array(image)
    input_batch = transform(image_array).to("cuda")

    timestart = time.time()
    with torch.no_grad():
        prediction = midas_model(input_batch)
    print("Time taken for depth map: ", time.time() - timestart)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image_array.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    depth_image = Image.fromarray((depth * 255 / np.max(depth)).astype(np.uint8)).convert("RGB")

    if (debug):
        depth_image.save("depth.jpg", format="JPEG")
    
    # Run controlnet
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    timestart = time.time()
    out_image = pipe(
        prompt, height = height, width=width, guidance_scale=guidance_scale ,num_inference_steps=steps, generator=generator, image=depth_image).images[0]
    print("Time taken for controlnet inference: ", time.time() - timestart)
    
    buffered = BytesIO()
    out_image.save(buffered, format="JPEG")

    if (debug):
        out_image.save("out.jpg", format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}

if __name__ == '__main__':
    init()
    with open("input/original.png", "rb") as f:
        img_bytes = f.read()

    inference({"prompt": "tropical room"}, img_bytes, debug=True)