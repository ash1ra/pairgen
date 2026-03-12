import numpy as np
import pytest
from PIL import Image

from pairgen import utils

TEST_IMRESIZE_PARAMETERS = [
    # img_height, img_width, scaling_factor
    [64, 64, 1 / 4],
    [64, 64, 4],
    [125, 61, 2],
    [64, 64, 3],
]

TEST_AUGMENTATIONS_PARAMETERS = [
    # img_height, img_width
    [64, 64],
    [128, 96],
    [63, 91],
]


def test_cubic() -> None:
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    y = utils.cubic(x)

    assert y.shape == x.shape
    assert not np.isnan(y).any()


def test_calculate_weights_indices() -> None:
    weights, indices = utils.calculate_weights_indices(
        in_length=100, out_length=50, scale=0.5, kernel_width=4, antialiasing=True
    )

    assert weights.shape[0] == 50
    assert indices.shape[0] == 50
    assert np.allclose(np.sum(weights, axis=1), 1.0)


def test_generate_gaussian_kernel() -> None:
    ksize = 21
    kernel = utils.generate_gaussian_kernel(ksize=ksize)

    assert kernel.shape == (ksize, ksize)
    assert np.isclose(np.sum(kernel), 1.0)


def test_generate_sinc_kernel() -> None:
    ksize = 15
    kernel = utils.generate_sinc_kernel(ksize=ksize)

    assert kernel.shape == (ksize, ksize)
    assert np.isclose(np.sum(kernel), 1.0)


def test_apply_kernel_fft() -> None:
    img = Image.new("RGB", (64, 64), color="blue")

    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1.0

    result_img = utils.apply_kernel_fft(img, kernel)

    assert isinstance(result_img, Image.Image)
    assert result_img.size == (64, 64)


@pytest.mark.parametrize("img_height, img_width, scaling_factor", TEST_IMRESIZE_PARAMETERS)
def test_imresize(img_height: int, img_width: int, scaling_factor: int) -> None:
    input_image_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    output_image_array = utils.imresize(input_image_array, scale=scaling_factor)

    assert output_image_array.shape == (img_height * scaling_factor, img_width * scaling_factor, 3)


def test_imresize_grayscale() -> None:
    input_image_array = np.zeros((64, 64), dtype=np.uint8)
    output_image_array = utils.imresize(input_image_array, scale=0.5)

    assert output_image_array.shape == (32, 32, 1)


@pytest.mark.parametrize("img_height, img_width", TEST_AUGMENTATIONS_PARAMETERS)
def test_apply_augmentations(img_height: int, img_width: int) -> None:
    input_img = Image.new("RGB", (img_height, img_width), color="green")
    output_img = utils.apply_augmentations(input_img)

    assert isinstance(output_img, Image.Image)
    assert output_img.size == (img_height, img_width) or output_img.size == (img_width, img_height)


def test_apply_degradations() -> None:
    input_img = Image.new("RGB", (64, 64), color="green")

    for _ in range(5):
        output_img = utils.apply_degradations(input_img, blur=True, noise=True, jpeg=True)
        assert isinstance(output_img, Image.Image)
        assert output_img.width > 0 and output_img.height > 0
