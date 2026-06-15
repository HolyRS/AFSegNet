import random
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_folder(root, candidates):
    for name in candidates:
        folder = root / name
        if folder.exists() and folder.is_dir():
            return folder
    raise FileNotFoundError(f"Cannot find folder from candidates: {candidates}")


def binarize_mask(mask_path):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    # If mask has multiple channels, convert it to a single-channel binary mask
    if len(mask.shape) == 3:
        mask = np.any(mask != 0, axis=2).astype(np.uint8)
    else:
        mask = (mask != 0).astype(np.uint8)

    return mask


def copy_pairs(pairs, output_root, split_name):
    image_out_dir = output_root / split_name / "images"
    mask_out_dir = output_root / split_name / "masks"

    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in tqdm(pairs, desc=f"Copying {split_name}"):
        # Copy image directly
        shutil.copy2(img_path, image_out_dir / img_path.name)

        # Convert mask non-zero values to 1, then save
        binary_mask = binarize_mask(mask_path)

        # Save mask with the original filename
        save_path = mask_out_dir / mask_path.name
        cv2.imwrite(str(save_path), binary_mask)


def split_and_copy_dataset(dataset_root, output_root, seed=42):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    image_dir = find_folder(
        dataset_root,
        ["images", "Images", "image", "Image", "JPEGImages"]
    )

    mask_dir = find_folder(
        dataset_root,
        ["masks", "Masks", "mask", "Mask", "labels", "Labels", "label", "Label"]
    )

    print(f"Image folder: {image_dir}")
    print(f"Mask folder:  {mask_dir}")

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    mask_files = [
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    mask_dict = {p.stem: p for p in mask_files}

    pairs = []
    missing_masks = []

    for img_path in image_files:
        stem = img_path.stem
        if stem in mask_dict:
            pairs.append((img_path, mask_dict[stem]))
        else:
            missing_masks.append(img_path.name)

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images have no matched masks.")
        print("Examples:", missing_masks[:5])

    if len(pairs) == 0:
        raise RuntimeError("No matched image-mask pairs found.")

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)

    train_num = int(total * 0.7)
    val_num = int(total * 0.15)

    train_pairs = pairs[:train_num]
    val_pairs = pairs[train_num:train_num + val_num]
    test_pairs = pairs[train_num + val_num:]

    print(f"Total matched samples: {total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")

    copy_pairs(train_pairs, output_root, "train")
    copy_pairs(val_pairs, output_root, "val")
    copy_pairs(test_pairs, output_root, "test")

    print(f"Done. Split dataset saved to: {output_root}")
    print("All mask values have been converted: non-zero -> 1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Abandoned Farmland 512",
        help="Path to the original dataset folder"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="Abandoned Farmland 512_split",
        help="Path to save the split dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    split_and_copy_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed
    )