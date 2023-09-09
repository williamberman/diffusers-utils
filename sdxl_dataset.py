import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms

from training_config import training_config


def get_sdxl_dataset():
    return (
        wds.WebDataset(training_config.train_shards, resampled=True)
        .shuffle(100)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(
            image="jpg;png;jpeg;webp",
            text="text;txt;caption",
            metadata="json",
            handler=wds.warn_and_continue,
        )
        .map(make_sample)
        .batched(
            training_config.batch_size, partial=False, collation_fn=default_collate
        )
    )


@torch.no_grad()
def make_sample(d):
    from sdxl import vae

    image = d["image"]
    text = d["text"]
    metadata = d["metadata"]

    resized_image = TF.resize(
        image,
        training_config.resolution,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(
        image, output_size=(training_config.resolution, training_config.resolution)
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

    latents = vae.encode(
        resized_and_cropped_and_normalized_image_tensor.to(
            device=vae.device, dtype=vae.dtype
        )
    ).latent_dist.sample()

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

    prompt_embeds, pooled_prompt_embeds_2 = text_conditioning(text)

    sample = {
        "time_ids": time_ids,
        "latents": latents.to("cpu"),
        "prompt_embeds": prompt_embeds.to("cpu"),
        "text_embeds": pooled_prompt_embeds_2.to("cpu"),
    }

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from mediapipe_pose import mediapipe_pose_adapter_image

            adapter_image = mediapipe_pose_adapter_image(
                resized_and_cropped_image.numpy()
            )
        else:
            assert False

        sample["adapter_image"] = adapter_image

    return sample


@torch.no_grad()
def text_conditioning(text):
    from sdxl import (text_encoder_one, text_encoder_two, tokenizer_one,
                      tokenizer_two)

    text_input_ids = tokenizer_one(
        text,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    prompt_embeds_1 = text_encoder_one(
        text_input_ids.to(dtype=text_encoder_one.dtype, device=text_encoder_one.device),
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_1 = prompt_embeds_1.view(
        prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1
    )

    text_input_ids = tokenizer_two(
        text,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    prompt_embeds_2 = text_encoder_two(
        text_input_ids.to(dtype=text_encoder_two.dtype, device=text_encoder_two.device),
        output_hidden_states=True,
    )

    pooled_prompt_embeds_2 = prompt_embeds_2[0]

    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds_2 = prompt_embeds_2.view(
        prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1
    )

    prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2))

    return prompt_embeds, pooled_prompt_embeds_2
