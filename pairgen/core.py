import concurrent.futures
import multiprocessing
import random
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import apply_augmentations, apply_degradations, imresize


def process_single_img(
    img_path: Path,
    hr_dir: Path,
    lr_dir: Path,
    scaling_factor: int,
    interpolation: str,
    patch_size: int,
    num_patches: int,
    augment: bool,
    blur: bool,
    noise: bool,
    jpeg: bool,
) -> None:
    try:
        with Image.open(img_path) as orig_img:
            orig_img = orig_img.convert("RGB")
            img_width, img_height = orig_img.size

            num_iterations = num_patches if patch_size > 0 else 1

            for i in range(num_iterations):
                suffix = f"_{i + 1:03d}" if patch_size > 0 else ""
                img_name = f"{img_path.stem}{suffix}.png"

                hr_img_path = hr_dir / img_name
                lr_img_path = lr_dir / img_name

                if hr_img_path.exists() and lr_img_path.exists():
                    if hr_img_path.stat().st_size > 0 and lr_img_path.stat().st_size > 0:
                        continue

                img = orig_img.copy()

                if patch_size > 0:
                    if img_width < patch_size or img_height < patch_size:
                        return

                    left = random.randint(0, img_width - patch_size)
                    top = random.randint(0, img_height - patch_size)
                    img = img.crop((left, top, left + patch_size, top + patch_size))
                else:
                    remainder_w = img_width % scaling_factor
                    remainder_h = img_height % scaling_factor

                    if remainder_w != 0 or remainder_h != 0:
                        img = img.crop((0, 0, img_width - remainder_w, img_height - remainder_h))

                if augment:
                    img = apply_augmentations(img)

                img.save(hr_img_path, compress_level=1)

                if interpolation == "matlab_bicubic":
                    lr_img_np = imresize(np.array(img), scale=1 / scaling_factor)
                    lr_img = Image.fromarray(lr_img_np)
                else:
                    new_width = img.size[0] // scaling_factor
                    new_height = img.size[1] // scaling_factor
                    lr_img = img.resize(
                        size=(new_width, new_height),
                        resample=getattr(Image.Resampling, interpolation.upper()),
                    )

                if blur or noise or jpeg:
                    lr_img = apply_degradations(img=lr_img, blur=blur, noise=noise, jpeg=jpeg)

                lr_img.save(lr_img_path, compress_level=1)
    except Exception as e:
        print(f"[Error] Failed to process '{img_path.name}': {e}")


def process_imgs(
    input_data_path: Path,
    output_data_path: Path,
    scaling_factor: int,
    recursive: bool = False,
    num_workers: int | None = None,
    interpolation: str = "matlab_bicubic",
    patch_size: int = 0,
    num_patches: int = 1,
    augment: bool = False,
    blur: bool = False,
    noise: bool = False,
    jpeg: bool = False,
) -> None:
    print(f"[Data] Preparing data from '{input_data_path}'...")

    output_data_path.mkdir(parents=True, exist_ok=True)

    hr_dir_output_path = output_data_path / "HR"
    hr_dir_output_path.mkdir(parents=True, exist_ok=True)

    lr_dir_output_path = output_data_path / f"LR_x{scaling_factor}"
    lr_dir_output_path.mkdir(parents=True, exist_ok=True)

    if input_data_path.exists():
        if input_data_path.is_dir():
            search_method = input_data_path.rglob("*") if recursive else input_data_path.glob("*")
            img_paths = sorted(
                [p for p in search_method if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
        elif input_data_path.is_file():
            with open(input_data_path, "r") as f:
                img_paths = sorted([Path(line.strip()) for line in f if line.strip()])
    else:
        raise FileNotFoundError(f"[Error] Input path '{input_data_path}' not found.")

    print(f"[Data] Found {len(img_paths)} images. Processing...")

    worker = partial(
        process_single_img,
        hr_dir=hr_dir_output_path,
        lr_dir=lr_dir_output_path,
        scaling_factor=scaling_factor,
        interpolation=interpolation,
        patch_size=patch_size,
        num_patches=num_patches,
        augment=augment,
        blur=blur,
        noise=noise,
        jpeg=jpeg,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        list(
            tqdm(
                executor.map(worker, img_paths),
                total=len(img_paths),
                desc="Processing images...",
                leave=False,
            )
        )

    print(f"[Data] Processing completed. Output saved to '{output_data_path}'.\n")
