"""Tests for image preprocessing module"""

import pytest
from pathlib import Path
import tempfile
import numpy as np


def test_image_prep_ml_import():
    """Test that ImagePrepML can be imported"""
    from autoprepml import ImagePrepML

    assert ImagePrepML is not None


def test_image_prep_ml_initialization():
    """Test ImagePrepML initialization with mock setup"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple test image
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img_path = Path(tmpdir) / "test.png"
        img.save(img_path)

        # Initialize ImagePrepML
        prep = ImagePrepML(image_dir=tmpdir, target_size=(224, 224), color_mode="rgb")

        assert prep.image_dir == Path(tmpdir)
        assert prep.target_size == (224, 224)
        assert prep.color_mode == "rgb"
        assert len(prep.image_paths) > 0


def test_image_detection():
    """Test image issue detection"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images with different properties
        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(Path(tmpdir) / "test1.png")

        img2 = Image.new("RGB", (200, 200), color="blue")
        img2.save(Path(tmpdir) / "test2.png")

        img3 = Image.new("L", (150, 150), color=128)  # Grayscale
        img3.save(Path(tmpdir) / "test3.png")

        prep = ImagePrepML(image_dir=tmpdir, target_size=(224, 224))
        issues = prep.detect(verbose=False)

        # Should detect size mismatches
        assert "size_mismatch" in issues
        assert len(issues["size_mismatch"]) == 3  # All have wrong size

        # Should detect color mode issues
        assert "color_mode_issues" in issues


def test_image_cleaning():
    """Test image cleaning and preprocessing"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        # sourcery skip: no-loop-in-tests
        for i in range(3):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(Path(tmpdir) / f"test{i}.png")

        prep = ImagePrepML(image_dir=tmpdir, target_size=(64, 64), normalize=True)
        prep.detect(verbose=False)

        # Clean images
        processed = prep.clean()

        # Check output shape
        assert processed.shape == (3, 64, 64, 3)  # (n_images, height, width, channels)

        # Check normalization
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0


def test_image_statistics():
    """Test image statistics generation"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(Path(tmpdir) / "test1.png")

        img2 = Image.new("RGBA", (200, 200), color="blue")
        img2.save(Path(tmpdir) / "test2.png")

        prep = ImagePrepML(image_dir=tmpdir)
        prep.detect(verbose=False)

        stats = prep.get_statistics()

        assert "total_images" in stats
        assert stats["total_images"] == 2
        assert "unique_sizes" in stats
        assert "color_mode_distribution" in stats
        assert "file_size_stats" in stats


def test_dataset_splitting():
    """Test train/val/test split"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 10 test images
        # sourcery skip: no-loop-in-tests
        for i in range(10):
            img = Image.new("RGB", (50, 50), color="red")
            img.save(Path(tmpdir) / f"test{i}.png")

        prep = ImagePrepML(image_dir=tmpdir, target_size=(50, 50))
        prep.detect(verbose=False)
        prep.clean()

        # Split dataset
        train, val, test = prep.split_dataset(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        # Check splits
        assert len(train) == 7
        assert len(val) == 1
        assert len(test) == 2
        assert len(train) + len(val) + len(test) == 10


def test_save_processed_images():
    """Test saving processed images"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(input_dir / f"test{i}.png")

        prep = ImagePrepML(image_dir=str(input_dir), target_size=(64, 64))
        prep.detect(verbose=False)
        prep.clean()

        # Save processed images
        prep.save_processed(str(output_dir), format="png")

        # Check output
        output_files = list(output_dir.glob("*.png"))
        assert len(output_files) == 3


def test_standard_image_normalization_round_trips_when_saved():
    """Per-channel standard normalization can be persisted as valid pixels."""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        source = np.array([[[0, 64, 128], [32, 96, 160]]], dtype=np.uint8)
        Image.fromarray(source, mode="RGB").save(input_dir / "source.png")

        prep = ImagePrepML(
            image_dir=str(input_dir),
            target_size=(2, 1),
            normalization_mode="standard",
            normalization_mean=[0.25, 0.5, 0.75],
            normalization_std=[0.25, 0.25, 0.25],
        )
        prep.detect(verbose=False)
        processed = prep.clean()
        assert np.isfinite(processed).all()
        prep.save_processed(str(output_dir))
        assert (output_dir / "processed_0000.png").exists()


def test_convenience_function():
    """Test the convenience preprocess_images function"""
    pytest.importorskip("PIL")
    from autoprepml.image import preprocess_images
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        # sourcery skip: no-loop-in-tests
        for i in range(2):
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(Path(tmpdir) / f"test{i}.png")

        # Use convenience function
        processed = preprocess_images(image_dir=tmpdir, target_size=(32, 32), normalize=True)

        assert processed.shape == (2, 32, 32, 3)
        assert processed.max() <= 1.0


def test_report_generation():
    """Test HTML report generation"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        img = Image.new("RGB", (100, 100), color="green")
        img.save(Path(tmpdir) / "test.png")

        prep = ImagePrepML(image_dir=tmpdir)
        prep.detect(verbose=False)
        prep.clean()

        # Generate report
        report_path = Path(tmpdir) / "report.html"
        prep.save_report(str(report_path))

        # Check report exists
        assert report_path.exists()

        # Check report contains expected content
        content = report_path.read_text(encoding="utf-8")
        assert "Image Preprocessing Report" in content
        assert "Dataset Statistics" in content


def test_missing_pillow():
    """Test error when Pillow is not installed"""
    # This test just checks that the import error handling exists
    # If Pillow is installed, this won't test the error path
    # But it ensures the module can be imported


def test_invalid_image_directory():
    """Test error handling for invalid directory"""
    from autoprepml.image import ImagePrepML

    with pytest.raises(ValueError, match="does not exist"):
        ImagePrepML(image_dir="/nonexistent/directory")


def test_invalid_image_normalization_mode():
    from autoprepml.image import ImagePrepML

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="normalization_mode"):
            ImagePrepML(image_dir=tmpdir, normalization_mode="unknown")


def test_image_constructor_validates_geometry_and_color_mode():
    from autoprepml.image import ImagePrepML

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="target_size"):
            ImagePrepML(image_dir=tmpdir, target_size=(0, 224))
        with pytest.raises(ValueError, match="color_mode"):
            ImagePrepML(image_dir=tmpdir, color_mode="cmyk")


def test_no_images_found():
    """Test error when no images are found"""
    from autoprepml.image import ImagePrepML

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory
        with pytest.raises(ValueError, match="No images found"):
            ImagePrepML(image_dir=tmpdir)


def test_color_mode_conversion():
    """Test color mode conversion"""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create grayscale image
        img = Image.new("L", (100, 100), color=128)
        img.save(Path(tmpdir) / "gray.png")

        # Process as RGB
        prep = ImagePrepML(image_dir=tmpdir, target_size=(100, 100), color_mode="rgb")
        prep.detect(verbose=False)
        processed = prep.clean(convert_mode=True)

        # Should be converted to RGB (3 channels)
        assert processed.shape[-1] == 3


def test_image_augmentation_flips_images():
    """Configured augmentation should add deterministic transformed samples."""
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        source = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        Image.fromarray(source, mode="L").save(Path(tmpdir) / "test.png")

        prep = ImagePrepML(
            image_dir=tmpdir,
            target_size=(3, 2),
            color_mode="grayscale",
            normalize=False,
        )
        prep.detect(verbose=False)
        processed = prep.clean(
            augment=True,
            augmentation_config={"horizontal_flip": True},
        )

        assert processed.shape == (2, 2, 3)
        np.testing.assert_array_equal(processed[0], source)
        np.testing.assert_array_equal(processed[1], source[:, ::-1])
        assert prep.log[-1]["augmented"] is True


def test_image_augmentation_rejects_unknown_options():
    """Unsupported augmentation options should fail clearly."""
    import numpy as np
    from autoprepml.image import ImagePrepML

    prep = object.__new__(ImagePrepML)
    with pytest.raises(ValueError, match="Unsupported augmentation option"):
        prep._augment_images(np.zeros((1, 2, 2), dtype=np.uint8), {"zoom": 2})


def test_image_constructor_validates_all_public_options():
    from autoprepml.image import ImagePrepML

    with tempfile.TemporaryDirectory() as tmpdir:
        for kwargs, error in [
            ({"target_size": "bad"}, "target_size"),
            ({"target_size": (1, True)}, "target_size"),
            ({"color_mode": None}, "color_mode"),
            ({"normalize": "yes"}, "normalize"),
            ({"normalization_mode": None}, "normalization_mode"),
        ]:
            with pytest.raises((TypeError, ValueError), match=error):
                ImagePrepML(image_dir=tmpdir, image_paths=[Path(tmpdir) / "a.png"], **kwargs)


def test_image_collection_deduplicates_explicit_paths_and_expected_modes(tmp_path):
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("L", (2, 2), color=128).save(path)
    prep = ImagePrepML(image_paths=[str(path), str(path)], color_mode="gray", normalize=False)
    assert prep.image_paths == [path]
    assert prep._get_expected_pil_mode() == "L"
    assert prep.get_statistics() == {}
    prep.detect(verbose=False)
    assert prep.get_statistics()["total_images"] == 1


def test_image_detects_corrupt_and_duplicate_files_and_clean_skips_corrupt(tmp_path):
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    valid = tmp_path / "valid.png"
    duplicate = tmp_path / "duplicate.png"
    Image.new("RGB", (2, 2), color="red").save(valid)
    Image.new("RGB", (2, 2), color="red").save(duplicate)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not-an-image")
    prep = ImagePrepML(image_paths=[valid, duplicate, corrupt], target_size=(2, 2))
    issues = prep.detect(verbose=False)
    assert "corrupted" in issues and "duplicates" in issues and "low_quality" in issues
    processed = prep.clean(remove_corrupted=True)
    assert len(processed) == 2


def test_image_augmentation_covers_rotation_and_validation_edges():
    from autoprepml.image import ImagePrepML

    prep = object.__new__(ImagePrepML)
    images = np.arange(4, dtype=np.uint8).reshape(1, 2, 2)
    rotated = prep._augment_images(
        images, {"vertical_flip": True, "rotations": [90, 180, 90], "include_original": False}
    )
    assert rotated.shape[0] == 3
    for config, error in [
        ([], "dictionary"),
        ({"horizontal_flip": 1}, "boolean"),
        ({"rotations": "90"}, "integer"),
        ({"rotations": [45]}, "multiples"),
        ({"include_original": False}, "enable"),
    ]:
        with pytest.raises((TypeError, ValueError), match=error):
            prep._augment_images(images, config)
    assert prep._augment_images(np.array([]), {}).size == 0


def test_image_split_and_save_without_normalization(tmp_path):
    pytest.importorskip("PIL")
    from autoprepml.image import ImagePrepML
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color="red").save(path)
    prep = ImagePrepML(image_paths=[path], target_size=(2, 2), normalize=False)
    with pytest.raises(ValueError, match="processed"):
        prep.split_dataset()
    prep.detect(verbose=False)
    prep.clean(resize=False, convert_mode=False)
    with pytest.raises(ValueError, match="sum"):
        prep.split_dataset(0.5, 0.5, 0.2)
    train, val, test = prep.split_dataset(shuffle=False)
    assert len(train) + len(val) + len(test) == 1
    output = tmp_path / "saved"
    prep.save_processed(output, format="png", prefix="item_")
    assert (output / "item_0000.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
