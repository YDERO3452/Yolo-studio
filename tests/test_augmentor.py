"""Tests for core/augmentor.py — DataAugmentor."""

import pytest
import numpy as np

from core.augmentor import DataAugmentor


class TestDataAugmentorInit:
    """Tests for DataAugmentor initialization."""

    def test_init(self):
        aug = DataAugmentor()
        assert aug.transform is None


class TestGetAvailableTransforms:
    """Tests for get_available_transforms static method."""

    def test_returns_dict(self):
        transforms = DataAugmentor.get_available_transforms()
        assert isinstance(transforms, dict)
        assert len(transforms) > 0

    def test_includes_common_transforms(self):
        transforms = DataAugmentor.get_available_transforms()
        assert "horizontal_flip" in transforms
        assert "random_brightness_contrast" in transforms
        assert "gaussian_noise" in transforms
        assert "blur" in transforms
        assert "rotate" in transforms


class TestAugmentImage:
    """Tests for augment_image method."""

    def test_returns_tuple(self):
        """augment_image returns (image, bboxes, labels) tuple."""
        aug = DataAugmentor()
        # Mock the transform to avoid albumentations dependency
        aug.transform = lambda **kwargs: {
            "image": kwargs["image"],
            "bboxes": kwargs["bboxes"],
            "class_labels": kwargs["class_labels"],
        }
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        labels = [0]
        result = aug.augment_image(image, bboxes, labels)
        assert len(result) == 3
        assert result[0].shape == (640, 640, 3)
        assert result[1] == bboxes
        assert result[2] == labels

    def test_fallback_on_error(self):
        """Returns original data when augmentation fails."""
        aug = DataAugmentor()
        aug.transform = lambda **kwargs: 1 / 0
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        labels = [0]
        result_img, result_bboxes, result_labels = aug.augment_image(image, bboxes, labels)
        np.testing.assert_array_equal(result_img, image)
        assert result_bboxes == bboxes
        assert result_labels == labels

    def test_empty_bboxes(self):
        """Works with empty bboxes."""
        aug = DataAugmentor()
        aug.transform = lambda **kwargs: {
            "image": kwargs["image"],
            "bboxes": [],
            "class_labels": [],
        }
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result_img, result_bboxes, result_labels = aug.augment_image(image, [], [])
        assert result_bboxes == []
        assert result_labels == []
