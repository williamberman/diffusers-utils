import math
import os
import random
from typing import Literal

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import xformers
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def maybe_ddp_dtype(m):
    if isinstance(m, DDP):
        m = m.module
    return m.dtype


def maybe_ddp_module(m):
    if isinstance(m, DDP):
        m = m.module
    return m


class ModelUtils:
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def load(cls, load_from, device):
        import load_state_dict_patch

        if os.path.isdir(load_from):
            load_from = os.path.join(load_from, "diffusion_pytorch_model.safetensors")

        with torch.device("meta"):
            model = cls()

        state_dict = safetensors.torch.load_file(load_from, device=device)

        model.load_state_dict(state_dict, assign=True)

        return model


def get_sinusoidal_embedding(
    indices: torch.Tensor,
    embedding_dim: int,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=indices.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = indices.unsqueeze(-1).float() * emb
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    return emb


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=None, eps=1e-5):
        super().__init__()

        if time_embedding_dim is not None:
            self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)
        else:
            self.time_emb_proj = None

        self.norm1 = torch.nn.GroupNorm(32, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=eps)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, hidden_states, temb=None):
        residual = hidden_states

        if self.time_emb_proj is not None:
            assert temb is not None
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual

        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim, num_transformer_blocks):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels, eps=1e-06)
        self.proj_in = nn.Linear(channels, channels)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, encoder_hidden_states_dim) for _ in range(num_transformer_blocks)])

        self.proj_out = nn.Linear(channels, channels)

    def forward(self, hidden_states, encoder_hidden_states):
        batch_size, channels, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

        hidden_states = hidden_states + residual

        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = Attention(channels, channels)

        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = Attention(channels, encoder_hidden_states_dim)

        self.norm3 = nn.LayerNorm(channels)
        self.ff = nn.ModuleDict(dict(net=nn.Sequential(GEGLU(channels, 4 * channels), nn.Dropout(0.0), nn.Linear(4 * channels, channels))))

    def forward(self, hidden_states, encoder_hidden_states):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states

        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states

        hidden_states = self.ff["net"](self.norm3(hidden_states)) + hidden_states

        return hidden_states


class Attention(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim, qkv_bias=False):
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_k = nn.Linear(encoder_hidden_states_dim, channels, bias=qkv_bias)
        self.to_v = nn.Linear(encoder_hidden_states_dim, channels, bias=qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.0))

    def forward(self, hidden_states, encoder_hidden_states=None):
        batch_size, q_seq_len, channels = hidden_states.shape
        head_dim = 64

        if encoder_hidden_states is not None:
            kv = encoder_hidden_states
        else:
            kv = hidden_states

        kv_seq_len = kv.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(kv)
        value = self.to_v(kv)

        query = query.reshape(batch_size, q_seq_len, channels // head_dim, head_dim).contiguous()
        key = key.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()
        value = value.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.reshape(batch_size, q_seq_len, channels).contiguous()

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


# General instructions: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
#
# Available models: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
#
# requires downloading model to root of repo i.e.
# `wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -O pose_landmarker.task`

mediapipe_pose_detector = None
_init_mediapipe_pose_called = False


def init_mediapipe_pose():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    global _init_mediapipe_pose_called, mediapipe_pose_detector

    if _init_mediapipe_pose_called:
        return

    _init_mediapipe_pose_called = True

    base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    mediapipe_pose_detector = vision.PoseLandmarker.create_from_options(options)


def mediapipe_pose_adapter_image(image, return_type: Literal["vae_scaled_tensor", "pil"] = "vae_scaled_tensor"):
    import mediapipe as mp

    init_mediapipe_pose()

    numpy_image = np.array(image)

    height, width = numpy_image.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    detection_result = mediapipe_pose_detector.detect(mp_image)

    pose_landmarks = detection_result.pose_landmarks

    if len(pose_landmarks) == 0:
        return None

    pose = np.zeros((height, width, 3), dtype=np.uint8)
    draw_landmarks_on_image(pose, pose_landmarks)

    if return_type == "vae_scaled_tensor":
        pose = torch.tensor(pose)
        pose = pose.permute(2, 0, 1)
        pose = pose.float()
        pose = pose / 255.0
    elif return_type == "pil":
        pose = Image.fromarray(pose)
    else:
        assert False

    return pose


def draw_landmarks_on_image(pose, pose_landmarks):
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    for idx in range(len(pose_landmarks)):
        pose_landmarks = pose_landmarks[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            pose,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )


# NOTE that this pil image cannot be used with the actual
# network because it uses 0 for the masked pixel insted of -1.
# It is used just for logging the masked image
def masked_image_as_pil(image: torch.Tensor) -> Image.Image:
    mask = image == -1
    image = image * (mask < 0.5)
    image = (image * 255).clamp(0, 255)
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    image = Image.fromarray(image)
    return image


def make_masked_image(image, return_type: Literal["controlnet_scaled_tensor", "vae_scaled_tensor"] = "controlnet_scaled_tensor", mask=None):
    if mask is None:
        mask = make_mask(image.height, image.width)

    mask = torch.from_numpy(mask)
    mask = mask[None, :, :]

    image = TF.to_tensor(image)

    if return_type == "controlnet_scaled_tensor":
        # where mask is set to 1, set to -1 "special" masked image pixel.
        # -1 is outside of the 0-1 range that the controlnet normalized
        # input is in.
        image = image * (mask < 0.5) + -1.0 * (mask > 0.5)
    elif return_type == "vae_scaled_tensor":
        # where mask is 1, zero out the pixels. Note that if you use the
        # image output of the `vae_scaled_tensor` such as in pre_encoded_controlnet_cond,
        # the network must also be passed the mask so it knows the zeroed out pixels are
        # from the mask and are not just zero in the original image
        image = image * (mask < 0.5)

        image = TF.normalize(image, [0.5], [0.5])
    else:
        assert False

    return image, mask


masking_types = ["full", "rectangle", "irregular", "outpainting"]


def make_mask(height, width):
    mask_type = random.choice(masking_types)

    if mask_type == "full":
        mask = np.ones((height, width), np.float32)
    elif mask_type == "rectangle":
        mask = make_random_rectangle_mask(height, width)
    elif mask_type == "irregular":
        mask = make_random_irregular_mask(height, width)
    elif mask_type == "outpainting":
        mask = make_outpainting_mask(height, width)
    else:
        assert False

    return mask


def make_random_rectangle_mask(
    height,
    width,
    margin=10,
    bbox_min_size=100,
    bbox_max_size=512,
    min_times=1,
    max_times=2,
):
    mask = np.zeros((height, width), np.float32)

    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)

        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)

        mask[start_y : start_y + box_height, start_x : start_x + box_width] = 1

    return mask


def make_random_irregular_mask(height, width, max_angle=4, max_len=60, max_width=256, min_times=1, max_times=2):
    import cv2

    mask = np.zeros((height, width), np.float32)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)

        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)

            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle

            length = 10 + np.random.randint(max_len)

            brush_w = 5 + np.random.randint(max_width)

            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)

            choice = random.randint(0, 2)

            if choice == 0:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif choice == 1:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1)
            elif choice == 2:
                radius = brush_w // 2
                mask[
                    start_y - radius : start_y + radius,
                    start_x - radius : start_x + radius,
                ] = 1
            else:
                assert False

            start_x, start_y = end_x, end_y

    return mask


def make_outpainting_mask(height, width, probs=[0.5, 0.5, 0.5, 0.5]):
    mask = np.zeros((height, width), np.float32)
    at_least_one_mask_applied = False

    coords = [
        [(0, 0), (1, get_padding(height))],
        [(0, 0), (get_padding(width), 1)],
        [(0, 1 - get_padding(height)), (1, 1)],
        [(1 - get_padding(width), 0), (1, 1)],
    ]

    for pp, coord in zip(probs, coords):
        if np.random.random() < pp:
            at_least_one_mask_applied = True
            mask = apply_padding(mask=mask, coord=coord)

    if not at_least_one_mask_applied:
        idx = np.random.choice(range(len(coords)), p=np.array(probs) / sum(probs))
        mask = apply_padding(mask=mask, coord=coords[idx])

    return mask


def get_padding(size, min_padding_percent=0.04, max_padding_percent=0.5):
    n1 = int(min_padding_percent * size)
    n2 = int(max_padding_percent * size)
    return np.random.randint(n1, n2) / size


def apply_padding(mask, coord):
    height, width = mask.shape

    mask[
        int(coord[0][0] * height) : int(coord[1][0] * height),
        int(coord[0][1] * width) : int(coord[1][1] * width),
    ] = 1

    return mask


open_pose = None
_init_openpose_called = False


def init_openpose():
    from controlnet_aux import OpenposeDetector

    global open_pose, _init_openpose_called

    if _init_openpose_called:
        return

    _init_openpose_called = True

    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def openpose_adapter_image(image, resolution, return_type: Literal["vae_scaled_tensor", "pil"] = "vae_scaled_tensor"):
    init_openpose()

    pose = open_pose(
        image,
        detect_resolution=resolution,
        image_resolution=resolution,
    )

    pose = np.array(pose)

    if (pose == 0).all():
        return None

    if return_type == "vae_scaled_tensor":
        pose = torch.tensor(pose)
        pose = pose.permute(2, 0, 1)
        pose = pose.float()
        pose = pose / 255.0
    elif return_type == "pil":
        pose = Image.fromarray(pose)
    else:
        assert False

    return pose


def make_canny_conditioning(
    image,
    return_type: Literal["controlnet_scaled_tensor", "pil"] = "controlnet_scaled_tensor",
):
    import cv2

    controlnet_image = np.array(image)
    controlnet_image = cv2.Canny(controlnet_image, 100, 200)
    controlnet_image = controlnet_image[:, :, None]
    controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)

    if return_type == "controlnet_scaled_tensor":
        controlnet_image = TF.to_tensor(controlnet_image)
    elif return_type == "pil":
        controlnet_image = Image.fromarray(controlnet_image)
    else:
        assert False

    return controlnet_image
