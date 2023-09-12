from controlnet_aux import OpenposeDetector
from typing import Literal
from training_config import training_config
import numpy as np
import torch
from PIL import Image

open_pose = None
_init_openpose_called = False

def init_openpose():
    global open_pose, _init_openpose_called

    if _init_openpose_called:
        return

    _init_openpose_called = True

    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

def openpose_adapter_image(image, return_type: Literal["vae_scaled_tensor", "pil"]='vae_scaled_tensor'):
    init_openpose()

    pose = open_pose(image, detect_resolution=training_config.resolution, image_resolution=training_config.resolution)

    pose = np.array(pose)

    if (pose == 0).all():
        return None

    if return_type == 'vae_scaled_tensor':
        pose = torch.tensor(pose)
        pose = pose.permute(2, 0, 1)
        pose = pose.float()
        pose = pose / 255.0
    elif return_type == 'pil':
        pose = Image.fromarray(pose)
    else:
        assert False

    return pose