import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import CLIPTokenizerFast

from training_config import training_config


def get_sdxl_dataset():
    return (
        wds.WebDataset(training_config.train_shards, resampled=True)
        .shuffle(training_config.shuffle_buffer_size)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(
            image="jpg;png;jpeg;webp",
            text="text;txt;caption",
            metadata="json",
            handler=wds.warn_and_continue,
        )
        .map(make_sample)
        .select(select)
        .batched(
            training_config.batch_size, partial=False, collation_fn=default_collate
        )
    )


tokenizer_one = CLIPTokenizerFast.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
)

tokenizer_two = CLIPTokenizerFast.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
)


@torch.no_grad()
def make_sample(d):
    image = d["image"]
    text = d["text"]
    metadata = d["metadata"]

    image = image.convert("RGB")

    resized_image = TF.resize(
        image,
        training_config.resolution,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(
        resized_image,
        output_size=(training_config.resolution, training_config.resolution),
    )

    resized_and_cropped_image = TF.crop(
        resized_image,
        c_top,
        c_left,
        training_config.resolution,
        training_config.resolution,
    )
    resized_and_cropped_image_tensor = TF.to_tensor(resized_and_cropped_image)
    resized_and_cropped_and_normalized_image_tensor = TF.normalize(
        resized_and_cropped_image_tensor, [0.5], [0.5]
    )

    original_width = int(metadata.get("original_width", 0.0))
    original_height = int(metadata.get("original_height", 0.0))

    time_ids = torch.tensor(
        [
            original_width,
            original_height,
            c_top,
            c_left,
            training_config.resolution,
            training_config.resolution,
        ]
    )

    text_input_ids_one = tokenizer_one(
        text,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    text_input_ids_two = tokenizer_two(
        text,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    sample = {
        "time_ids": time_ids,
        "text_input_ids_one": text_input_ids_one,
        "text_input_ids_two": text_input_ids_two,
        "image": resized_and_cropped_and_normalized_image_tensor,
    }

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from mediapipe_pose import mediapipe_pose_adapter_image

            adapter_image = mediapipe_pose_adapter_image(
                np.array(resized_and_cropped_image)
            )

            sample["adapter_image"] = adapter_image
        else:
            assert False
    else:
        assert False

    return sample


def select(sample):
    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            return sample["adapter_image"] is not None
        else:
            assert False

    else:
        assert False
