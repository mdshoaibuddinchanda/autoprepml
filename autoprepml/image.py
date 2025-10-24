"""Image data preprocessing module for AutoPrepML"""
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import json


class ImagePrepML:
    """Image data preprocessing class.
    
    Handles common image preprocessing tasks:
    - Loading and validation
    - Resizing and normalization
    - Augmentation
    - Format conversion
    - Quality checks
    - Batch processing
    
    Example:
        >>> prep = ImagePrepML(image_dir='./images', target_size=(224, 224))
        >>> prep.detect()
        >>> processed_images = prep.clean()
        >>> prep.save_report('image_report.html')
    """
    
    def __init__(
        self,
        image_dir: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (224, 224),
        color_mode: str = 'rgb',
        normalize: bool = True
    ):
        """Initialize ImagePrepML.
        
        Args:
            image_dir: Directory containing images
            image_paths: List of specific image paths
            target_size: Target image dimensions (height, width)
            color_mode: Color mode ('rgb', 'grayscale', 'rgba')
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_paths = image_paths or []
        self.target_size = target_size
        self.color_mode = color_mode.lower()
        self.normalize = normalize
        
        self.log = []
        self.issues = {}
        self.images = []
        self.image_info = []
        
        # Validation
        if self.image_dir and not self.image_dir.exists():
            raise ValueError(f"Image directory '{image_dir}' does not exist")
        
        # Supported formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Collect image paths
        self._collect_images()
        
    def _collect_images(self):
        """Collect all image paths from directory or provided list"""
        collected_paths = []
        
        if self.image_dir:
            for fmt in self.supported_formats:
                collected_paths.extend(self.image_dir.glob(f"*{fmt}"))
                collected_paths.extend(self.image_dir.glob(f"*{fmt.upper()}"))
        
        # Use existing paths or collected ones
        if self.image_paths:
            self.image_paths = [Path(p) for p in self.image_paths]
        else:
            self.image_paths = [Path(p) for p in collected_paths]
        
        # Remove duplicates
        self.image_paths = list(set(self.image_paths))
        
        if not self.image_paths:
            raise ValueError("No images found. Provide image_dir or image_paths.")
        
        self.log.append({
            "action": "initialized",
            "image_count": len(self.image_paths),
            "target_size": self.target_size,
            "color_mode": self.color_mode
        })
    
    def detect(self, verbose: bool = True) -> Dict[str, Any]:
        """Detect issues in images.
        
        Args:
            verbose: Print detection results
            
        Returns:
            Dictionary with detected issues
        """
        try:
            import PIL
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for image preprocessing.\n"
                "Install with: pip install Pillow"
            ) from e

        issues = {
            'corrupted': [],
            'wrong_format': [],
            'size_mismatch': [],
            'color_mode_issues': [],
            'low_quality': [],
            'duplicates': []
        }

        sizes = {}
        color_modes = {}
        file_hashes = {}

        for img_path in self.image_paths:
            try:
                # Try to open image
                img = Image.open(img_path)

                # Verify image
                img.verify()

                # Re-open for processing (verify closes the file)
                img = Image.open(img_path)

                # Get info
                info = {
                    'path': str(img_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': img_path.stat().st_size
                }
                self.image_info.append(info)

                # Check size
                if img.size != self.target_size:
                    sizes[img.size] = sizes.get(img.size, 0) + 1
                    issues['size_mismatch'].append({
                        'path': str(img_path),
                        'current': img.size,
                        'target': self.target_size
                    })

                # Check color mode
                expected_mode = self._get_expected_pil_mode()
                if img.mode != expected_mode:
                    color_modes[img.mode] = color_modes.get(img.mode, 0) + 1
                    issues['color_mode_issues'].append({
                        'path': str(img_path),
                        'current': img.mode,
                        'expected': expected_mode
                    })

                # Check for duplicates (simple hash-based)
                img_hash = hash(img.tobytes())
                if img_hash in file_hashes:
                    issues['duplicates'].append({
                        'path': str(img_path),
                        'duplicate_of': file_hashes[img_hash]
                    })
                else:
                    file_hashes[img_hash] = str(img_path)

                # Check quality (file size)
                if info['file_size'] < 1024:  # Less than 1KB
                    issues['low_quality'].append({
                        'path': str(img_path),
                        'file_size': info['file_size']
                    })

                img.close()

            except (IOError, OSError, PIL.UnidentifiedImageError) as e:
                issues['corrupted'].append({
                    'path': str(img_path),
                    'error': str(e)
                })
            except Exception as e:
                issues['wrong_format'].append({
                    'path': str(img_path),
                    'error': str(e)
                })

        # Filter out empty issue categories
        self.issues = {k: v for k, v in issues.items() if v}

        # Log detection
        self.log.append({
            "action": "detection_complete",
            "total_images": len(self.image_paths),
            "valid_images": len(self.image_info),
            "issues_found": len(self.issues),
            "issue_summary": {k: len(v) for k, v in self.issues.items()}
        })

        if verbose:
            self._print_detection_summary()

        return self.issues
    
    def _get_expected_pil_mode(self) -> str:
        """Get expected PIL image mode"""
        mode_map = {
            'rgb': 'RGB',
            'rgba': 'RGBA',
            'grayscale': 'L',
            'gray': 'L'
        }
        return mode_map.get(self.color_mode, 'RGB')
    
    def _print_detection_summary(self):
        """Print detection summary"""
        print("\n" + "="*80)
        print("IMAGE PREPROCESSING - DETECTION SUMMARY")
        print("="*80)
        print(f"\n📁 Total Images: {len(self.image_paths)}")
        print(f"✅ Valid Images: {len(self.image_info)}")
        print(f"⚠️  Issues Found: {len(self.issues)}")
        
        if self.issues:
            print("\n📋 Issue Breakdown:")
            for issue_type, items in self.issues.items():
                print(f"   • {issue_type}: {len(items)}")
        else:
            print("\n✨ No issues detected!")
    
    def clean(
        self,
        remove_corrupted: bool = True,
        resize: bool = True,
        convert_mode: bool = True,
        augment: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Clean and preprocess images.
        
        Args:
            remove_corrupted: Remove corrupted images
            resize: Resize images to target size
            convert_mode: Convert color mode
            augment: Apply data augmentation
            augmentation_config: Augmentation parameters
            
        Returns:
            Numpy array of processed images
        """
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required. Install: pip install Pillow") from e

        processed = []
        corrupted_paths = {item['path'] for item in self.issues.get('corrupted', [])}

        for img_path in self.image_paths:
            # Skip corrupted if requested
            if remove_corrupted and str(img_path) in corrupted_paths:
                continue

            try:
                img = Image.open(img_path)

                # Convert color mode if needed
                if convert_mode:
                    expected_mode = self._get_expected_pil_mode()
                    if img.mode != expected_mode:
                        img = img.convert(expected_mode)

                # Resize if needed
                if resize and img.size != self.target_size:
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)

                # Convert to numpy array
                img_array = np.array(img)

                # Normalize if requested
                if self.normalize:
                    img_array = img_array.astype(np.float32) / 255.0

                processed.append(img_array)
                img.close()

            except Exception as e:
                self.log.append({
                    "action": "processing_error",
                    "path": str(img_path),
                    "error": str(e)
                })
                continue

        # Convert to numpy array
        processed_array = np.array(processed)

        # Apply augmentation if requested
        if augment and augmentation_config:
            processed_array = self._augment_images(processed_array, augmentation_config)

        self.images = processed_array

        # Log cleaning
        self.log.append({
            "action": "cleaning_complete",
            "processed_count": len(processed),
            "output_shape": processed_array.shape,
            "normalized": self.normalize,
            "augmented": augment
        })

        return processed_array
    
    def _augment_images(
        self,
        images: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply data augmentation.
        
        Args:
            images: Array of images
            config: Augmentation configuration
            
        Returns:
            Augmented images
        """
        # This would use libraries like imgaug, albumentations, or tf.keras
        # For now, return original images
        # TODO: Implement augmentation with optional dependencies
        return images
    
    def save_processed(
        self,
        output_dir: str,
        format: str = 'png',
        prefix: str = 'processed_'
    ):
        """Save processed images to directory.
        
        Args:
            output_dir: Output directory path
            format: Image format (png, jpg, etc.)
            prefix: Filename prefix
        """
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required. Install: pip install Pillow") from e

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, img_array in enumerate(self.images):
            # Denormalize if needed
            if self.normalize:
                img_array = (img_array * 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array)

            # Save
            output_file = output_path / f"{prefix}{i:04d}.{format}"
            img.save(output_file)

        self.log.append({
            "action": "images_saved",
            "output_dir": str(output_path),
            "count": len(self.images),
            "format": format
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the image dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.image_info:
            return {}

        sizes = [info['size'] for info in self.image_info]
        modes = [info['mode'] for info in self.image_info]
        file_sizes = [info['file_size'] for info in self.image_info]

        # Count occurrences
        from collections import Counter
        size_counts = Counter(sizes)
        mode_counts = Counter(modes)

        return {
            'total_images': len(self.image_info),
            'unique_sizes': len(size_counts),
            'size_distribution': dict(size_counts.most_common(5)),
            'color_mode_distribution': dict(mode_counts),
            'file_size_stats': {
                'min_bytes': min(file_sizes),
                'max_bytes': max(file_sizes),
                'avg_bytes': sum(file_sizes) / len(file_sizes),
                'total_mb': sum(file_sizes) / (1024 * 1024),
            },
        }
    
    def save_report(self, output_path: str = 'image_report.html'):
        """Generate and save HTML report.
        
        Args:
            output_path: Path to save the HTML report
        """
        stats = self.get_statistics()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Image Preprocessing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; font-size: 14px; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
        .issue-card {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .success {{ background: #d4edda; border-left-color: #28a745; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .log-entry {{ font-family: monospace; background: #2c3e50; color: #ecf0f1; padding: 8px; margin: 5px 0; border-radius: 4px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Image Preprocessing Report</h1>
        
        <h2>📊 Dataset Statistics</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="value">{stats.get('total_images', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Unique Sizes</h3>
                <div class="value">{stats.get('unique_sizes', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Size</h3>
                <div class="value">{stats.get('file_size_stats', {}).get('total_mb', 0):.2f} MB</div>
            </div>
            <div class="stat-card">
                <h3>Avg File Size</h3>
                <div class="value">{stats.get('file_size_stats', {}).get('avg_bytes', 0) / 1024:.1f} KB</div>
            </div>
        </div>
        
        <h2>🎨 Color Mode Distribution</h2>
        <table>
            <tr><th>Mode</th><th>Count</th></tr>
            {''.join(f'<tr><td>{mode}</td><td>{count}</td></tr>' for mode, count in stats.get('color_mode_distribution', {}).items())}
        </table>
        
        <h2>⚠️ Issues Detected</h2>
        {self._generate_issues_html()}
        
        <h2>📝 Processing Log</h2>
        {''.join(f'<div class="log-entry">{json.dumps(entry, indent=2)}</div>' for entry in self.log)}
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ Report saved to: {output_path}")
    
    def _generate_issues_html(self) -> str:
        """Generate HTML for issues section"""
        if not self.issues:
            return '<div class="issue-card success">✨ No issues detected! All images are valid.</div>'

        return "".join(
            f'<div class="issue-card"><strong>{issue_type.upper()}</strong>: {len(items)} items</div>'
            for issue_type, items in self.issues.items()
        )
    
    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_state: Optional[int] = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Shuffle before splitting
            random_state: Random seed
            
        Returns:
            Tuple of (train, val, test) arrays
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        if len(self.images) == 0:
            raise ValueError("No processed images. Call clean() first.")
        
        n = len(self.images)
        indices = np.arange(n)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return (
            self.images[train_idx],
            self.images[val_idx],
            self.images[test_idx]
        )


# Convenience function
def preprocess_images(
    image_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    color_mode: str = 'rgb',
    normalize: bool = True,
    output_dir: Optional[str] = None
) -> np.ndarray:
    """Quick image preprocessing function.
    
    Args:
        image_dir: Directory containing images
        target_size: Target dimensions
        color_mode: Color mode
        normalize: Normalize pixels
        output_dir: Save processed images here
        
    Returns:
        Processed image array
    """
    prep = ImagePrepML(
        image_dir=image_dir,
        target_size=target_size,
        color_mode=color_mode,
        normalize=normalize
    )
    prep.detect()
    images = prep.clean()
    
    if output_dir:
        prep.save_processed(output_dir)
    
    return images
