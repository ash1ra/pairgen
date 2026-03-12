from pathlib import Path

import pytest
from PIL import Image

from pairgen.core import process_imgs, process_single_img

TEST_PARAMETERS = [
    # img_height, img_width, scaling_factor
    [64, 64, 2],
    [64, 64, 3],
    [64, 64, 4],
    [71, 121, 2],
    [71, 121, 3],
]


@pytest.mark.parametrize("img_height, img_width, scaling_factor", TEST_PARAMETERS)
def test_process_single_img(
    img_height: int,
    img_width: int,
    scaling_factor: int,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x4"

    input_dir.mkdir()
    hr_dir.mkdir()
    lr_dir.mkdir()

    img_path = input_dir / "image.png"
    Image.new("RGB", (img_height, img_width), color="green").save(img_path)

    process_single_img(
        img_path=img_path,
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scaling_factor=scaling_factor,
        resampling_method="bicubic",
        patch_size=0,
        num_patches=1,
        augment=False,
        blur=False,
        noise=False,
        jpeg=False,
    )

    expected_hr_path = hr_dir / "image.png"
    expected_lr_path = lr_dir / "image.png"

    assert expected_hr_path.exists()
    assert expected_lr_path.exists()

    hr_img = Image.open(expected_hr_path)
    lr_img = Image.open(expected_lr_path)

    expected_lr_img_height = img_height // scaling_factor
    expected_lr_img_width = img_width // scaling_factor

    expected_hr_img_height = expected_lr_img_height * scaling_factor
    expected_hr_img_width = expected_lr_img_width * scaling_factor

    assert hr_img.size == (expected_hr_img_height, expected_hr_img_width)
    assert lr_img.size == (expected_lr_img_height, expected_lr_img_width)


@pytest.mark.parametrize("img_height, img_width, scaling_factor", TEST_PARAMETERS)
def test_process_single_img_patches(
    img_height: int,
    img_width: int,
    scaling_factor: int,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x4"

    input_dir.mkdir()
    hr_dir.mkdir()
    lr_dir.mkdir()

    img_path = input_dir / "image.png"
    Image.new("RGB", (img_height, img_width), color="green").save(img_path)

    patch_size = 64

    process_single_img(
        img_path=img_path,
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scaling_factor=scaling_factor,
        resampling_method="bicubic",
        patch_size=patch_size,
        num_patches=3,
        augment=False,
        blur=False,
        noise=False,
        jpeg=False,
    )

    expected_lr_img_height = patch_size // scaling_factor
    expected_lr_img_width = patch_size // scaling_factor

    expected_hr_imgs = list(hr_dir.glob("*.png"))
    assert len(expected_hr_imgs) == 3

    for hr_img in expected_hr_imgs:
        assert Image.open(hr_img).size == (patch_size, patch_size)

    expected_lr_imgs = list(lr_dir.glob("*.png"))
    assert len(expected_lr_imgs) == 3

    for lr_img in expected_lr_imgs:
        assert Image.open(lr_img).size == (expected_lr_img_height, expected_lr_img_width)


def test_process_single_img_skip_existing(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x4"

    input_dir.mkdir()
    hr_dir.mkdir()
    lr_dir.mkdir()

    img_path = input_dir / "image.png"
    Image.new("RGB", (64, 64), color="green").save(img_path)

    hr_img_path = hr_dir / "image.png"
    lr_img_path = lr_dir / "image.png"
    Image.new("RGB", (64, 64), color="red").save(hr_img_path)
    Image.new("RGB", (32, 32), color="red").save(lr_img_path)

    process_single_img(
        img_path=img_path,
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scaling_factor=2,
        resampling_method="bicubic",
        patch_size=0,
        num_patches=1,
        augment=False,
        blur=False,
        noise=False,
        jpeg=False,
    )

    assert Image.open(hr_img_path).getpixel((0, 0)) == (255, 0, 0)


def test_process_imgs_with_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    Image.new("RGB", (64, 64)).save(input_dir / "1.png")
    Image.new("RGB", (64, 64)).save(input_dir / "2.jpg")

    process_imgs(input_data_path=input_dir, output_data_path=output_dir, scaling_factor=2, num_workers=1)

    assert (output_dir / "HR" / "1.png").exists()
    assert (output_dir / "LR_x2" / "2.png").exists()


def test_process_imgs_with_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    input_img_1 = input_dir / "1.png"
    input_img_2 = input_dir / "2.png"
    Image.new("RGB", (64, 64)).save(input_img_1)
    Image.new("RGB", (64, 64)).save(input_img_2)

    file = tmp_path / "file.txt"
    with open(file, "w") as f:
        f.write(f"{input_img_1}\n{input_img_2}\n")

    process_imgs(input_data_path=file, output_data_path=output_dir, scaling_factor=2, num_workers=1)

    assert (output_dir / "HR" / "1.png").exists()
    assert (output_dir / "LR_x2" / "2.png").exists()
