import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPTokenizerFast


def maybe_ddp_dtype(m):
    if isinstance(m, DDP):
        m = m.module
    return m.dtype


def maybe_ddp_module(m):
    if isinstance(m, DDP):
        m = m.module
    return m


# TODO: would be nice to just call a function from a tokenizers https://github.com/huggingface/tokenizers
# i.e. afaik tokenizing shouldn't require holding any state

tokenizer_one = CLIPTokenizerFast.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")

tokenizer_two = CLIPTokenizerFast.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")


def sdxl_tokenize_one(prompts):
    return tokenizer_one(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]


def sdxl_tokenize_two(prompts):
    return tokenizer_two(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]


def sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two):
    prompt_embeds_1 = text_encoder_one(
        text_input_ids_one,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_1 = prompt_embeds_1.view(prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1)

    prompt_embeds_2 = text_encoder_two(
        text_input_ids_two,
        output_hidden_states=True,
    )

    pooled_encoder_hidden_states = prompt_embeds_2[0]

    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds_2 = prompt_embeds_2.view(prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1)

    encoder_hidden_states = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)

    return encoder_hidden_states, pooled_encoder_hidden_states


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
