"""Data augmentation module using Albumentations."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from loguru import logger


class DataAugmentor:
    """Provides data augmentation for YOLO training datasets."""

    def __init__(self):
        self.transform = None

    def build_transform(self, config=None) -> "albumentations.Compose":
        """Build an Albumentations transform pipeline."""
        import albumentations as A

        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(std_range=(0.2, 0.44), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomSizedBBoxSafeCrop(height=640, width=640, p=0.3),
        ]

        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_area=100,
                min_visibility=0.3,
            ),
        )
        return self.transform

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> tuple:
        """Apply augmentation to a single image with bboxes."""
        if self.transform is None:
            self.build_transform()

        try:
            result = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            return result["image"], result["bboxes"], result["class_labels"]
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original")
            return image, bboxes, class_labels

    def augment_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        augmentations_per_image: int = 2,
    ):
        """Augment an entire dataset."""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        out_images = Path(output_images_dir)
        out_labels = Path(output_labels_dir)

        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]

        if self.transform is None:
            self.build_transform()

        total_augmented = 0
        for img_file in image_files:
            label_file = labels_path / (img_file.stem + ".txt")

            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read labels
            bboxes = []
            class_labels = []
            if label_file.exists():
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            bboxes.append(coords)
                            class_labels.append(class_id)

            # Copy original
            cv2.imwrite(str(out_images / img_file.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if label_file.exists():
                import shutil
                shutil.copy2(label_file, out_labels / label_file.name)

            # Generate augmented versions
            for i in range(augmentations_per_image):
                try:
                    aug_image, aug_bboxes, aug_labels = self.augment_image(image, bboxes, class_labels)

                    aug_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                    cv2.imwrite(str(out_images / aug_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                    # Write augmented labels
                    with open(out_labels / f"{img_file.stem}_aug{i}.txt", "w", encoding="utf-8") as f:
                        for bbox, cls in zip(aug_bboxes, aug_labels):
                            f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

                    total_augmented += 1
                except Exception as e:
                    logger.warning(f"Failed to augment {img_file.name} #{i}: {e}")

        logger.info(f"Augmentation complete: {total_augmented} new images generated")
        return total_augmented

    def preview_augmentation(
        self,
        image_path: str,
        label_path: str,
        num_samples: int = 4,
    ) -> list[np.ndarray]:
        """Preview augmentations on a single image."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])

        if self.transform is None:
            self.build_transform()

        samples = []
        for _ in range(num_samples):
            aug_img, _, _ = self.augment_image(image, bboxes, class_labels)
            samples.append(aug_img)

        return samples

    @staticmethod
    def get_available_transforms() -> dict:
        """List available augmentation transforms."""
        return {
            "horizontal_flip": "水平翻转",
            "vertical_flip": "垂直翻转",
            "random_brightness_contrast": "随机亮度对比度",
            "hue_saturation": "色调饱和度",
            "gaussian_noise": "高斯噪声",
            "blur": "模糊",
            "rotate": "旋转",
            "scale": "缩放",
            "translate": "平移",
            "mosaic": "马赛克",
            "mixup": "混合",
            "cutout": "随机遮挡",
        }
