"""
Image Preprocessing Demo - AutoPrepML
Demonstrates image data preprocessing capabilities
"""

import numpy as np
from pathlib import Path
import tempfile

print("=" * 80)
print("AutoPrepML - Image Preprocessing Demo")
print("=" * 80)

# Check if PIL is installed
try:
    from PIL import Image
    from autoprepml import ImagePrepML
except ImportError as e:
    print("\n❌ Error: Pillow (PIL) is required for image preprocessing")
    print("   Install with: pip install Pillow")
    print(f"\n   Error details: {e}")
    exit(1)

print("\n✅ Required libraries loaded successfully")

# Create sample images for demonstration
print("\n" + "=" * 80)
print("Step 1: Creating Sample Dataset")
print("=" * 80)

# Create temporary directory with sample images
tmpdir = tempfile.mkdtemp()
image_dir = Path(tmpdir) / 'sample_images'
image_dir.mkdir(parents=True, exist_ok=True)

# Create diverse sample images
print(f"\nCreating sample images in: {image_dir}")

# Different sizes
sizes = [(100, 100), (200, 200), (150, 150), (100, 100), (300, 200)]
colors = ['red', 'blue', 'green', 'yellow', 'purple']
modes = ['RGB', 'RGB', 'RGB', 'L', 'RGBA']  # Include grayscale and RGBA

for i, (size, color, mode) in enumerate(zip(sizes, colors, modes)):
    if mode == 'L':
        img = Image.new(mode, size, color=128)
    elif mode == 'RGBA':
        img = Image.new(mode, size, color=(255, 0, 255, 255))
    else:
        img = Image.new(mode, size, color=color)
    
    img.save(image_dir / f'sample_{i}.png')
    print(f"   ✓ Created sample_{i}.png ({size[0]}x{size[1]}, {mode})")

print(f"\n📁 Total images created: {len(list(image_dir.glob('*.png')))}")

# Initialize ImagePrepML
print("\n" + "=" * 80)
print("Step 2: Initialize Image Preprocessing")
print("=" * 80)

prep = ImagePrepML(
    image_dir=str(image_dir),
    target_size=(224, 224),
    color_mode='rgb',
    normalize=True
)

print(f"\n✅ ImagePrepML initialized")
print(f"   • Target size: {prep.target_size}")
print(f"   • Color mode: {prep.color_mode}")
print(f"   • Normalize: {prep.normalize}")
print(f"   • Images found: {len(prep.image_paths)}")

# Detect issues
print("\n" + "=" * 80)
print("Step 3: Detect Image Issues")
print("=" * 80)

issues = prep.detect(verbose=True)

# Clean and preprocess
print("\n" + "=" * 80)
print("Step 4: Clean and Preprocess Images")
print("=" * 80)

processed_images = prep.clean(
    remove_corrupted=True,
    resize=True,
    convert_mode=True
)

print(f"\n✅ Preprocessing complete!")
print(f"   • Processed images shape: {processed_images.shape}")
print(f"   • Data type: {processed_images.dtype}")
print(f"   • Value range: [{processed_images.min():.3f}, {processed_images.max():.3f}]")
print(f"   • Memory usage: {processed_images.nbytes / 1024 / 1024:.2f} MB")

# Get statistics
print("\n" + "=" * 80)
print("Step 5: Dataset Statistics")
print("=" * 80)

stats = prep.get_statistics()

print(f"\n📊 Dataset Overview:")
print(f"   • Total images: {stats['total_images']}")
print(f"   • Unique sizes: {stats['unique_sizes']}")
print(f"   • Total dataset size: {stats['file_size_stats']['total_mb']:.2f} MB")
print(f"   • Average file size: {stats['file_size_stats']['avg_bytes'] / 1024:.1f} KB")

print(f"\n🎨 Color Mode Distribution:")
for mode, count in stats['color_mode_distribution'].items():
    print(f"   • {mode}: {count} images")

print(f"\n📏 Size Distribution (Top 5):")
for size, count in list(stats['size_distribution'].items())[:5]:
    print(f"   • {size}: {count} images")

# Split dataset
print("\n" + "=" * 80)
print("Step 6: Split Dataset (Train/Val/Test)")
print("=" * 80)

train, val, test = prep.split_dataset(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    shuffle=True,
    random_state=42
)

print(f"\n✅ Dataset split complete:")
print(f"   • Training set: {len(train)} images ({len(train)/len(processed_images)*100:.1f}%)")
print(f"   • Validation set: {len(val)} images ({len(val)/len(processed_images)*100:.1f}%)")
print(f"   • Test set: {len(test)} images ({len(test)/len(processed_images)*100:.1f}%)")
print(f"   • Total: {len(train) + len(val) + len(test)} images")

# Save processed images
print("\n" + "=" * 80)
print("Step 7: Save Processed Images")
print("=" * 80)

output_dir = Path(tmpdir) / 'processed'
prep.save_processed(str(output_dir), format='png', prefix='clean_')

saved_files = list(output_dir.glob('*.png'))
print(f"\n✅ Saved {len(saved_files)} processed images to: {output_dir}")
for f in saved_files[:3]:
    print(f"   • {f.name}")
if len(saved_files) > 3:
    print(f"   • ... and {len(saved_files) - 3} more")

# Generate report
print("\n" + "=" * 80)
print("Step 8: Generate HTML Report")
print("=" * 80)

report_path = Path(tmpdir) / 'image_report.html'
prep.save_report(str(report_path))

print(f"\n✅ HTML report generated: {report_path}")
print("   Open this file in a browser to view detailed analysis")

# Demonstrate convenience function
print("\n" + "=" * 80)
print("Step 9: Quick Preprocessing (Convenience Function)")
print("=" * 80)

from autoprepml.image import preprocess_images

quick_processed = preprocess_images(
    image_dir=str(image_dir),
    target_size=(128, 128),
    color_mode='rgb',
    normalize=True
)

print(f"\n✅ Quick preprocessing complete:")
print(f"   • Output shape: {quick_processed.shape}")
print("   • One-line function call!")

# Summary
print("\n" + "=" * 80)
print("✨ Demo Complete - Summary")
print("=" * 80)

print(f"""
📝 Key Features Demonstrated:

1. ✅ Image Loading & Validation
   - Automatic format detection
   - Corruption checking
   - Size/mode validation

2. ✅ Issue Detection
   - Size mismatches: {len(issues.get('size_mismatch', []))}
   - Color mode issues: {len(issues.get('color_mode_issues', []))}
   - Corrupted images: {len(issues.get('corrupted', []))}

3. ✅ Preprocessing Pipeline
   - Resizing to target dimensions
   - Color mode conversion
   - Normalization to [0, 1]

4. ✅ Dataset Management
   - Statistics generation
   - Train/val/test splitting
   - Batch processing

5. ✅ Output Generation
   - Processed image saving
   - HTML reporting
   - Numpy array export

💡 Usage Patterns:

   # Full control
   prep = ImagePrepML(image_dir='./data')
   prep.detect()
   images = prep.clean()
   
   # Quick processing
   images = preprocess_images('./data', target_size=(224, 224))

📚 Supported Formats: {', '.join(prep.supported_formats)}

🎯 Perfect for:
   - ML model training data preparation
   - Image quality assessment
   - Batch image processing
   - Dataset standardization
""")

print("\n" + "=" * 80)
print("📁 Output Files (in temporary directory):")
print("=" * 80)
print(f"\n   {tmpdir}")
print("   ├── sample_images/          (original images)")
print("   ├── processed/               (cleaned images)")
print("   └── image_report.html        (detailed report)")

print("\n💡 Tip: Check the HTML report for detailed visualization!")
print("\n" + "=" * 80)
print("🎉 Demo finished successfully!")
print("=" * 80)

# Cleanup note
print(f"\n📌 Note: Temporary files created in: {tmpdir}")
print("   These will be cleaned up automatically on system restart")
print("   Or manually delete the directory when done")
