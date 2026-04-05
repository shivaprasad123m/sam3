import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_stitched_comparison(image_paths, output_dir):
    """Create 3x3 stitched grid showing original image, bboxes, segmentation, and uncertainty comparison"""

    for i, image_path in enumerate(image_paths, 1):
        base_name = f"result_{i}"
        
        # Load the actual original image from the source directory
        image_name = "groceries.jpg" if i == 1 else "truck.jpg"
        actual_original_path = os.path.join(os.path.dirname(output_dir), "assets", "uncertainImages", image_name)
        actual_original = Image.open(actual_original_path)

        # Load all processed images
        original_inference_path = os.path.join(output_dir, f"{base_name}_original_inference.png")
        original_segmentation_path = os.path.join(output_dir, f"{base_name}_original_segmentation.png")
        mc_bboxes_path = os.path.join(output_dir, f"{base_name}.png")  # MC-Dropout with BBoxes
        uncertainty_map_path = os.path.join(output_dir, f"{base_name}_uncertainty_map.png")
        mc_segmentation_path = os.path.join(output_dir, f"{base_name}_mc_segmentation.png")

        original_inference = Image.open(original_inference_path)
        original_segmentation = Image.open(original_segmentation_path)
        mc_bboxes = Image.open(mc_bboxes_path)
        uncertainty_map = Image.open(uncertainty_map_path)
        mc_segmentation = Image.open(mc_segmentation_path)

        # Create 3x3 figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # Row 1: Original Results
        # [0,0] Original Image
        axes[0, 0].imshow(actual_original)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # [0,1] Original SAM3 with BBoxes
        axes[0, 1].imshow(original_inference)
        axes[0, 1].set_title('Original SAM3\n(BBoxes)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # [0,2] Original SAM3 Segmentation Masks
        axes[0, 2].imshow(original_segmentation)
        axes[0, 2].set_title('Original SAM3\n(Segmentation)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: MC-Dropout Uncertainty Results
        # [1,0] MC-Dropout with BBoxes
        axes[1, 0].imshow(mc_bboxes)
        axes[1, 0].set_title('MC-Dropout\n(BBoxes)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # [1,1] Uncertainty Heatmap
        axes[1, 1].imshow(uncertainty_map)
        axes[1, 1].set_title('Uncertainty Map\n(MC-Dropout)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # [1,2] MC-Dropout Segmentation Masks
        axes[1, 2].imshow(mc_segmentation)
        axes[1, 2].set_title('MC-Dropout\n(Segmentation)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        # Row 3: Empty - for clean look
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

        # Set overall title
        fig.suptitle(f'SAM3 3x3 Analysis Grid - {image_name}', fontsize=18, fontweight='bold', y=0.995)

        # Save stitched image
        stitched_path = os.path.join(output_dir, f"stitched_comparison_3x3_{i}_{image_name.replace('.jpg', '')}.png")
        plt.tight_layout()
        plt.savefig(stitched_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"Created 3x3 stitched comparison: {stitched_path}")

if __name__ == "__main__":
    output_dir = "inference_results_uncertainty"
    image_paths = ["groceries.jpg", "truck.jpg"]

    create_stitched_comparison(image_paths, output_dir)