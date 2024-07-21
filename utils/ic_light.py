import os
import math
import random
import string
import numpy as np
import torch
import safetensors.torch as sf
import albumentations as A
import cv2
from diffusers.utils import load_image

from PIL import Image, ImageFilter, ImageOps
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionLatentUpscalePipeline,
)
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from enum import Enum

# from torch.hub import download_url_to_file


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = "stablediffusionapi/realistic-vision-v51"
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
)

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8,
        unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding,
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs["cross_attention_kwargs"] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = "./models/iclight_sd15_fc.safetensors"
# download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)
sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device("cuda")
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1,
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None,
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None,
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [
        [id_start] + tokens[i : i + chunk_length] + [id_end]
        for i in range(0, len(tokens), chunk_length)
    ]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = (
        torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    )  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def remove_alpha_threshold(image, alpha_threshold=160):
    # This function removes artifacts created by LayerDiffusion
    mask = image[:, :, 3] < alpha_threshold
    image[mask] = [0, 0, 0, 0]
    return image


@torch.inference_mode()
def process(
    input_fg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
    lowres_denoise,
    bg_source,
):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise "Wrong initial latent!"

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    if input_bg is None:
        latents = (
            t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = (
            i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [
        resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64),
        )
        for p in pixels
    ]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    latents = (
        i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


def augment(image):

    original = image.copy()

    image_height, image_width, _ = original.shape

    if random.choice([True, False]):
        target_height, target_width = 640 * 2, 512 * 2
    else:
        target_height, target_width = 512 * 2, 640 * 2

    left_right_padding = (
        max(target_width, image_width) - min(target_width, image_width)
    ) // 2

    original = cv2.copyMakeBorder(
        original,
        top=max(target_height, image_height) - min(target_height, image_height),
        bottom=0,
        left=left_right_padding,
        right=left_right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit_x=(-0.2, 0.2),
                shift_limit_y=(0.0, 0.2),
                scale_limit=(0, 0),
                rotate_limit=(-2, 2),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ]
    )

    return transform(image=original)["image"]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


input_dir = "hdf5/images"
output_dir = "dataset"
ground_truth_dir = os.path.join(output_dir, "gr")
image_dir = os.path.join(output_dir, "im")

prompts = [
    "sunshine, cafe, chilled, morning light",
    "exhibition, paintings, soft spotlight",
    "beach, bright sunlight",
    "winter, snow, overcast",
    "forest, cloudy, diffused light",
    "party, people, colorful lights",
    "cozy living room, sofa, shelf, warm lamplight",
    "mountains, dawn light",
    "nature, landscape, golden hour",
    "city centre, busy, evening lights",
    "neighbourhood, street, cars, streetlights",
    "bright sun from behind, sunset, dark shadows",
    "apartment, soft light, afternoon glow",
    "garden, dappled sunlight",
    "school, fluorescent lights",
    "art exhibition with paintings in background, gallery lighting",
    "library, reading, soft overhead light",
    "concert, stage, vibrant lighting",
    "office, workspace, natural daylight",
    "restaurant, dinner, candlelight",
    "park, playground, midday sun",
    "market, stalls, string lights",
    "train station, commuters, artificial light",
    "theater, audience, dimmed lights",
    "museum, artifacts, controlled lighting",
    "gym, workout, bright ceiling lights",
    "hotel lobby, reception, ambient lighting",
    "hospital, corridor, clinical lights",
    "airport, terminal, morning light",
    "sports field, night game, floodlights",
    "nightclub, dancing, strobe lights",
    "zoo, animals, soft daylight",
    "library, study area, natural light",
    "subway, platform, underground lighting",
    "restaurant, breakfast, morning light",
    "forest trail, hikers, filtered sunlight",
    "desert, dunes, harsh sunlight",
    "waterfall, mist, morning light",
    "city park, joggers, early morning",
    "beach, sunset, golden light",
    "festival, crowd, colorful lights",
    "skyscraper, office, evening light",
    "backyard, family, twilight",
    "coffee shop, reading, cozy lighting",
    "boutique, shopping, bright lights",
    "gymnasium, basketball, bright lighting",
    "street market, vendors, evening light",
    "wedding, ceremony, soft lighting",
    "farm, barn, morning light",
    "harbor, boats, dawn light",
    "rainy day, window, natural light",
    "mountain cabin, fireplace, warm glow",
    "playground, children, bright daylight",
    "skating rink, ice, artificial lighting",
    "beach, bonfire, nighttime",
    "forest, camping, moonlight",
    "garden party, guests, string lights",
    "university campus, students, midday sun",
    "amusement park, rides, vibrant lighting",
    "castle, tourists, afternoon light",
    "ski resort, slopes, bright snow reflection",
    "riverbank, fishing, early morning light",
    "suburban street, night, streetlights",
]

os.makedirs(ground_truth_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

all_images = os.listdir(input_dir)
random.shuffle(all_images)

for filename in all_images:
    if filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):  # Check if the file is an image

        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for i in range(13))
        random_filename = f"{random_string}_{filename}"

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mask = image[:, :, 3] < 100
        image[mask] = [0, 0, 0, 0]

        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = np.array(image)

        image_augmented = augment(image)
        Image.fromarray(image_augmented).getchannel("A").save(
            os.path.join(ground_truth_dir, random_filename)
        )

        image_augmented = image_augmented[:, :, :3]

        # We half the size and width because SD 1.5 creates much better results then
        image_augmented = image_augmented[::2, ::2]
        image_height, image_width, _ = image_augmented.shape

        num_samples = 1
        seed = random.randint(1, 123456789012345678901234567890)
        steps = 25
        constant_prompt = "details, high quality"
        prompt = random.choice(prompts)
        n_prompt = "bad quality, blurry"
        cfg = 2.0
        highres_scale = 2.0
        highres_denoise = 0.7
        lowres_denoise = 0.5
        bg_source = BGSource.NONE

        results = process(
            image_augmented,
            constant_prompt,
            image_width,
            image_height,
            num_samples,
            seed,
            steps,
            prompt,
            n_prompt,
            cfg,
            highres_scale,
            highres_denoise,
            lowres_denoise,
            bg_source,
        )
        result_image = Image.fromarray(results[0])
        result_image.save(os.path.join(image_dir, random_filename))
