# Ported from BasicSR (matlab_functions.py) to ensure academic reproducibility.
# Implements MATLAB-like bicubic interpolation required for standard SR benchmarks (Set5, Set14).
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py
import io
import random

import numpy as np
from PIL import Image


def cubic(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3

    return (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * ((abs_x <= 1).astype(x.dtype)) + (
        -0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2
    ) * (((abs_x > 1) * (abs_x <= 2)).astype(x.dtype))


def calculate_weights_indices(
    in_length: int,
    out_length: int,
    scale: float,
    kernel_width: int,
    antialiasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if (scale < 1) and antialiasing:
        kernel_width: int | float = kernel_width / scale

    x = np.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2

    indices = left.reshape(int(out_length), 1) + np.linspace(0, p - 1, p).reshape(1, int(p))

    distance_to_center = u.reshape(int(out_length), 1) - indices

    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    weights_sum = np.sum(weights, 1).reshape(int(out_length), 1)
    weights /= weights_sum

    weights_zero_idx = np.where(weights_sum == 0)
    if len(weights_zero_idx[0]) > 0:
        weights[weights_zero_idx, 0] = 1

    padded_indices = indices.astype(int)
    padded_indices -= 1

    padded_indices = np.abs(padded_indices)
    padded_indices = np.where(padded_indices < in_length, padded_indices, 2 * in_length - 1 - padded_indices)
    padded_indices = np.clip(padded_indices, 0, in_length - 1)

    return weights, padded_indices


def imresize(img: np.ndarray, scale: float, antialiasing: bool = True) -> np.ndarray:
    if scale == 1:
        return img

    if len(img.shape) == 3:
        input_img_height, input_img_width, input_img_num_channels = img.shape
    else:
        input_img_height, input_img_width = img.shape
        input_img_num_channels = 1

    output_img_height = int(np.ceil(input_img_height * scale))
    output_img_width = int(np.ceil(input_img_width * scale))

    kernel_width = 4

    height_weights, height_indices = calculate_weights_indices(
        in_length=input_img_height,
        out_length=output_img_height,
        scale=scale,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    width_weights, width_indices = calculate_weights_indices(
        in_length=input_img_width,
        out_length=output_img_width,
        scale=scale,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    img_aug = np.zeros((output_img_height, input_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img[:, :, channel] if input_img_num_channels > 1 else img
        pixels = channel_data[height_indices]
        img_aug[:, :, channel] = np.sum(height_weights[:, :, None] * pixels, axis=1)

    output_img = np.zeros((output_img_height, output_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img_aug[:, :, channel]
        pixels = channel_data[:, width_indices]
        output_img[:, :, channel] = np.sum(width_weights[None, :, :] * pixels, axis=2)

    output_img = np.clip(output_img, 0, 255)

    return np.round(output_img).astype(np.uint8)


def apply_augmentations(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    rotation = random.choice(
        [
            None,
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        ]
    )
    if rotation is not None:
        img = img.transpose(rotation)

    return img


def apply_kernel_fft(img: Image.Image, kernel: np.ndarray) -> Image.Image:
    img_np = np.array(img).astype(np.float32)
    H, W, C = img_np.shape
    K = kernel.shape[0]

    pad_h, pad_w = H - K, W - K
    padded_kernel = np.pad(kernel, ((0, pad_h), (0, pad_w)), mode="constant")

    padded_kernel = np.roll(padded_kernel, shift=(-K // 2, -K // 2), axis=(0, 1))

    kernel_fft = np.fft.fft2(padded_kernel)[:, :, np.newaxis]
    img_fft = np.fft.fft2(img_np, axes=(0, 1))

    result = np.real(np.fft.ifft2(img_fft * kernel_fft, axes=(0, 1)))

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_gaussian_kernel(ksize=21):
    sigma_x = random.uniform(0.2, 3.0)
    sigma_y = random.uniform(0.2, sigma_x)
    angle = random.uniform(0, np.pi)

    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = xx * cos_a - yy * sin_a
    yr = xx * sin_a + yy * cos_a

    kernel = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
    return kernel / np.sum(kernel)


def generate_sinc_kernel(ksize=21):
    cutoff = random.uniform(0.3, 0.8)
    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    r = np.sqrt(xx**2 + yy**2)
    r = np.maximum(r, 1e-8)

    kernel = np.sin(2 * np.pi * cutoff * r) / (2 * np.pi * cutoff * r)
    kernel[ksize // 2, ksize // 2] = 1.0
    return kernel / np.sum(kernel)


def apply_blur(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        kernel = generate_gaussian_kernel(ksize=random.choice([7, 9, 11, 15, 21]))
    else:
        kernel = generate_sinc_kernel(ksize=random.choice([11, 15, 21]))
    return apply_kernel_fft(img, kernel)


def apply_noise(img: Image.Image) -> Image.Image:
    img_np = np.array(img).astype(np.float32)

    if random.random() < 0.5:
        sigma = random.uniform(1.0, 15.0)
        noise = np.random.normal(0, sigma, img_np.shape)
        noisy_img = img_np + noise
    else:
        scale = random.uniform(0.05, 3.0)
        noisy_img = np.random.poisson(img_np * scale) / scale

    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))


def apply_jpeg(img: Image.Image) -> Image.Image:
    quality = random.randint(30, 95)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def apply_random_resize(img: Image.Image) -> Image.Image:
    scale = random.uniform(0.8, 1.2)
    w, h = img.size
    method = random.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
    return img.resize((int(w * scale), int(h * scale)), resample=method).resize((w, h), resample=method)


def apply_degradations(
    img: Image.Image,
    blur: bool = False,
    noise: bool = False,
    jpeg: bool = False,
) -> Image.Image:
    degradation_pipeline = []

    if blur:
        degradation_pipeline.append(apply_blur)

    if noise:
        degradation_pipeline.append(apply_noise)

    if jpeg:
        degradation_pipeline.append(apply_jpeg)

    if degradation_pipeline:
        degradation_pipeline.append(apply_random_resize)

    random.shuffle(degradation_pipeline)

    for degradation in degradation_pipeline:
        if random.random() < 0.8:
            img = degradation(img)

    return img
