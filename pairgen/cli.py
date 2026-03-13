import argparse
from pathlib import Path

from .core import process_imgs


def main() -> None:
    parser = argparse.ArgumentParser(description="pairgen — fast and reliable HR-LR image pair generator.")

    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Input directory or manifest file to scan.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory where HR and LR folders will be created.",
    )
    parser.add_argument(
        "-s",
        "--scaling-factor",
        type=int,
        required=True,
        help="Scaling factor for LR images.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for strict sequential order (default: 1).",
    )
    parser.add_argument(
        "-im",
        "--interpolation",
        type=str,
        choices=["matlab_bicubic", "bilinear", "bicubic", "lanczos", "nearest"],
        default="matlab_bicubic",
        help="Interpolation method (default: matlab_bicubic).",
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        type=int,
        default=0,
        help="If > 0, extracts a random square patch of this size from HR.",
    )
    parser.add_argument(
        "-np",
        "--num-patches",
        type=int,
        default=1,
        help="Number of random patches to extract per image (default: 1).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply random flips and rotations to HR.",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Apply random Gaussian blur to LR.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply random noise to LR.",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Apply random JPEG compression to LR.",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input path does not exist: '{args.input_path}'.")

    if args.workers < 1:
        parser.error(f"Worker count must be >= 1. Got: {args.workers}.")

    process_imgs(
        input_data_path=args.input_path,
        output_data_path=args.output_dir,
        scaling_factor=args.scaling_factor,
        recursive=args.recursive,
        num_workers=args.workers,
        interpolation=args.interpolation,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        augment=args.augment,
        blur=args.blur,
        noise=args.noise,
        jpeg=args.jpeg,
    )


if __name__ == "__main__":
    main()
