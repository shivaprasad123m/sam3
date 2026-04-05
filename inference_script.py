import os

from create_stitched_images import create_stitched_comparison
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import login
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS

# Set up torch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def plot_and_save_results(img, results, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    object_uncertainties = results.get("object_uncertainties", [0.0] * nb_objects)
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        mask_tensor = results["masks"][i]
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        plot_mask(mask_tensor.cpu(), ax, color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        uncertainty = object_uncertainties[i]
        if "boxes" in results and results["boxes"] is not None and len(results["boxes"]) > i:
            plot_bbox(ax, h, w, results["boxes"][i].cpu(), text=None, box_format="XYXY", color=color, relative_coords=False)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_mask(mask, ax, color):
    mask = mask.cpu().numpy()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color + (0.6,)).reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_segmentation_masks_only(img, results, save_path):
    """Save only segmentation masks without bboxes - overlaid on original image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    nb_objects = len(results["scores"])
    
    # Create a composite mask with all objects
    img_array = np.asarray(img)
    h, w = img_array.shape[:2]
    
    # Draw all masks on the image
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        mask_tensor = results["masks"][i]
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        
        # Get mask as numpy array
        mask_np = mask_tensor.cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        
        # Normalize mask to 0-1
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        
        # Create colored overlay
        mask_colored = np.zeros((h, w, 4), dtype=np.float32)
        mask_colored[..., :3] = np.array(color) / 255.0 if max(color) > 1 else np.array(color)
        mask_colored[..., 3] = mask_np * 0.6  # 60% alpha
        
        # Overlay on image
        ax.imshow(mask_colored, alpha=0.6)
    
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_bbox(ax, img_height, img_width, bbox, text, box_format, color, relative_coords):
    bbox_arr = np.asarray(bbox)
    if bbox_arr.ndim != 1 or bbox_arr.shape[0] != 4:
        raise ValueError(f"Unsupported bbox shape: {bbox_arr.shape}")

    if box_format == "XYXY":
        x1, y1, x2, y2 = bbox_arr.astype(float)
    elif box_format == "CXCYWH":
        cx, cy, w, h = bbox_arr.astype(float)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    else:
        raise ValueError(f"Unsupported box format: {box_format}")

    if relative_coords or np.max(bbox_arr) <= 1.0:
        x1 *= img_width
        x2 *= img_width
        y1 *= img_height
        y2 *= img_height

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor=color, facecolor='none', zorder=5)
    ax.add_patch(rect)


def enable_mc_dropout(model):
    """Enable dropout layers for MC dropout inference - only in decoder layers."""
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            # Only enable dropout in decoder-related layers to reduce variability
            if 'decoder' in name.lower() or 'mask' in name.lower():
                module.train()
                dropout_count += 1
    print(f"  Enabled MC-dropout in {dropout_count} decoder/mask layers")
    return dropout_count


def save_uncertainty_map(mask_std, save_path, original_image=None, result_state=None):
    """Save uncertainty map overlaid on original image as heatmap with high-confidence bounding boxes."""
    fig, ax = plt.subplots(figsize=(12, 8))

    if isinstance(mask_std, torch.Tensor):
        mask_std_np = mask_std.detach().cpu().numpy()
    else:
        mask_std_np = np.asarray(mask_std)

    # Display original image as background
    if original_image is not None:
        original_image_np = np.asarray(original_image)
        ax.imshow(original_image_np, alpha=0.6)  # Increased alpha for better visibility

    # Overlay uncertainty map with hot colormap - use lower alpha
    uncertainty_map = mask_std_np.copy()
    # Only show uncertainty above a threshold to avoid cluttering
    threshold = np.percentile(uncertainty_map, 75)  # Show top 25% uncertainty
    uncertainty_map[uncertainty_map < threshold] = 0

    if np.max(uncertainty_map) > 0:
        im = ax.imshow(uncertainty_map, cmap='hot', interpolation='bilinear', alpha=0.4, vmin=0, vmax=np.max(uncertainty_map))
    
    def draw_bbox_on_ax(ax, box, img_height, img_width, color='lime', text=None):
        box_arr = np.asarray(box, dtype=float).reshape(-1)
        if box_arr.size != 4:
            return
        if np.max(box_arr) <= 1.0 and np.min(box_arr) >= 0.0:
            x1, y1, x2, y2 = box_arr * np.array([img_width, img_height, img_width, img_height])
        else:
            x1, y1, x2, y2 = box_arr.tolist()
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=6,  # Thicker lines
            edgecolor=color,
            facecolor='none',
            zorder=20,  # Higher z-order
        )
        ax.add_patch(rect)
    
    if result_state is not None:
        boxes = result_state.get("boxes")
        object_uncertainties = result_state.get("object_uncertainties", [])
        
        print(f"  Debug: boxes={len(boxes) if boxes is not None else 0}, uncertainties={len(object_uncertainties)}")
        
        if boxes is not None and len(boxes) > 0:
            img_h = original_image_np.shape[0] if original_image is not None else None
            img_w = original_image_np.shape[1] if original_image is not None else None
            
            for i, box in enumerate(boxes):
                if isinstance(box, torch.Tensor):
                    box = box.detach().cpu().numpy().astype(float)
                
                # Get uncertainty value or 0 if not available
                uncertainty = float(object_uncertainties[i]) if i < len(object_uncertainties) else 0.0
                
                draw_bbox_on_ax(
                    ax,
                    box,
                    img_h,
                    img_w,
                    color='lime',
                    text=None,
                )
    
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    print(f"  Saved uncertainty map to {save_path}")



def original_inference(processor, image, text_prompt='bags'):
    """Run original SAM3 inference without MC-dropout for baseline comparison."""
    processor.model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=text_prompt, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")

    if masks is None or scores is None or len(scores) == 0:
        return None

    # Create result state similar to MC-dropout version
    result_state = {
        "masks": masks,
        "scores": scores,
        "boxes": boxes,
        "object_uncertainties": [0.0] * len(scores)  # No uncertainty for original
    }

    return result_state


def mc_dropout_inference(processor, image, num_runs=3, text_prompt='bags'):
    """Per-image stochastic inference runs for uncertainty estimation using a text prompt for small object detection."""
    width, height = image.size
    mask_logits_list = []
    score_list = []
    states = []

    for run_id in range(num_runs):
        processor.model.eval()
        dropout_count = enable_mc_dropout(processor.model)

        with torch.no_grad(): 
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=text_prompt, state=state)

        mask_logits = state.get("masks_logits")
        scores = state.get("scores")

        if mask_logits is None or scores is None or len(scores) == 0:
            print(f"    Run {run_id+1}: No valid prediction")
            continue

        print(f"    Run {run_id+1}: Got {len(scores)} objects, score={scores[0].item():.4f}")
        mask_logits_list.append(mask_logits.detach().cpu().numpy())
        score_list.append(scores.detach().cpu().numpy())
        states.append(state)

    if len(mask_logits_list) == 0:
        return None, None, None

    # Stack along runs: shape (num_runs, max_objects, H, W) but need to handle varying num_objects
    # For simplicity, since we use pixel-level uncertainty, we'll compute std across runs for each pixel
    # But to handle multiple objects, we'll take the max across objects per run per pixel
    max_objects = max(len(m) for m in mask_logits_list)
    H, W = mask_logits_list[0].shape[-2:] if len(mask_logits_list[0]) > 0 else (height, width)

    # Create a stack of shape (num_runs, H, W) taking max logit across objects per run
    logits_stack = np.zeros((len(mask_logits_list), H, W))
    for r, logits in enumerate(mask_logits_list):
        if len(logits) > 0:
            logits_stack[r] = np.max(logits, axis=0)  # max across objects

    # Convert logits to probabilities
    probabilities_stack = 1 / (1 + np.exp(-logits_stack))
    mean_mask = np.mean(probabilities_stack, axis=0)
    std_mask = np.std(probabilities_stack, axis=0)

    print(f"  Debug: mask_logits shape={logits_stack.shape}, std_mask min={std_mask.min():.6f}, max={std_mask.max():.6f}, mean={std_mask.mean():.6f}")

    # For scores, take the max score per run
    scores_arr = np.array([np.max(s) if len(s) > 0 else 0 for s in score_list])
    score_mean = float(np.mean(scores_arr))
    score_std = float(np.std(scores_arr))

    stats = {
        "score_mean": score_mean,
        "score_std": score_std,
        "mask_mean": mean_mask,
        "mask_std": std_mask,
    }

    return stats, states[-1], logits_stack

def main():
    # Paths
    sam3_root = os.path.dirname(os.path.abspath(__file__))
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    image_dir = os.path.join(sam3_root, "assets", "uncertainImages")
    output_dir = os.path.join(sam3_root, "inference_results_uncertainty")
    os.makedirs(output_dir, exist_ok=True)

    # Authenticate with HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        # os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        # login(token=hf_token)
        print(f"✓ Skipping login (model assumed cached)")

    # Build model
    model = build_sam3_image_model(bpe_path=bpe_path, device="cuda")

    # Convert model to float32 for CPU compatibility
    model = model.to(dtype=torch.float32)
    
    # Count dropout layers for verification
    num_dropouts = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    print(f"✓ Model has {num_dropouts} Dropout layers")

    # Get all image files from uncertainImages folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    image_paths = [os.path.join(image_dir, f) for f in sorted(image_files)]

    print(f"Found {len(image_paths)} images in {image_dir}")

    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        image = Image.open(image_path)

        processor = Sam3Processor(model, confidence_threshold=0.1, device="cuda")

        # MC-dropout inference (5 runs) for uncertainty estimation using text prompt detection
        # Use appropriate prompt based on image content
        if 'truck' in image_path.lower():
            text_prompt = 'truck'
        elif 'grocery' in image_path.lower() or 'groceries' in image_path.lower():
            text_prompt = 'bags'
        else:
            text_prompt = 'objects'  # fallback
        
        stats, result_state, _ = mc_dropout_inference(processor, image, num_runs=20, text_prompt=text_prompt)

        if stats is None or result_state is None:
            print(f"  ✗ No valid prediction for {image_path}")
            continue

        # Filter out empty masks and compute per-object uncertainties
        masks = result_state.get("masks")
        scores = result_state.get("scores")
        boxes = result_state.get("boxes")
        
        if masks is None or len(masks) == 0:
            print(f"  ✗ No masks found for {image_path}")
            continue
            
        valid_indices = []
        object_uncertainties = []
        mask_std = stats['mask_std']
        
        for i, mask in enumerate(masks):
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            # Check if mask has any pixels
            if np.sum(mask_np > 0) == 0:
                continue
            # Compute uncertainty as mean std within mask
            uncertainty = np.mean(mask_std[mask_np > 0])
            valid_indices.append(i)
            object_uncertainties.append(uncertainty)
        
        if len(valid_indices) == 0:
            print(f"  ✗ No valid objects found for {image_path}")
            continue
            
        # Filter results to only valid objects
        filtered_masks = [masks[i] for i in valid_indices]
        filtered_scores = [scores[i] for i in valid_indices] if scores is not None else None
        filtered_boxes = [boxes[i] for i in valid_indices] if boxes is not None else None
        
        # Update result_state with filtered results (all valid objects, no high-confidence filtering)
        result_state["masks"] = filtered_masks
        result_state["scores"] = filtered_scores
        result_state["boxes"] = filtered_boxes
        result_state["object_uncertainties"] = object_uncertainties

        print(
            f"  ✓ Score: mean={stats['score_mean']:.4f}, std={stats['score_std']:.4f}, {len(object_uncertainties)} total valid objects"
        )

        # Save segmentation result
        result_name = f"result_{idx+1}"
        save_path = os.path.join(output_dir, f"{result_name}.png")
        plot_and_save_results(image, result_state, save_path)

        # Save uncertainty statistics
        uncertainty_text = f"score_mean={stats['score_mean']:.3f}, score_std={stats['score_std']:.3f}\n"
        uncertainty_text += f"mask_std min={stats['mask_std'].min():.4f}, max={stats['mask_std'].max():.4f}, mean={stats['mask_std'].mean():.4f}\n"
        object_uncertainties = result_state.get("object_uncertainties", [])
        if object_uncertainties:
            uncertainty_text += f"per-object uncertainties: {', '.join(f'{u:.3f}' for u in object_uncertainties)}"
        with open(save_path + ".uncertainty.txt", "w") as f:
            f.write(uncertainty_text + "\n")
        print(f"  Saved stats to {save_path}.uncertainty.txt")

        # Save uncertainty map heatmap
        uncertainty_map_path = os.path.join(output_dir, f"{result_name}_uncertainty_map.png")
        save_uncertainty_map(stats['mask_std'].squeeze(), uncertainty_map_path, original_image=image, result_state=result_state)

        # Run original inference for comparison
        print(f"  Running original inference for comparison...")
        original_result_state = original_inference(processor, image, text_prompt=text_prompt)

        if original_result_state is not None:
            # Save original inference result with bboxes
            original_result_path = os.path.join(output_dir, f"{result_name}_original_inference.png")
            plot_and_save_results(image, original_result_state, original_result_path)
            print(f"  Saved original inference to {original_result_path}")
            
            # Save original inference segmentation masks only (no bboxes)
            original_segmentation_path = os.path.join(output_dir, f"{result_name}_original_segmentation.png")
            save_segmentation_masks_only(image, original_result_state, original_segmentation_path)
            print(f"  Saved original segmentation to {original_segmentation_path}")
        else:
            print(f"  ✗ Original inference failed for {image_path}")
        
        # Save MC-dropout segmentation masks only (no bboxes)
        mc_segmentation_path = os.path.join(output_dir, f"{result_name}_mc_segmentation.png")
        save_segmentation_masks_only(image, result_state, mc_segmentation_path)
        print(f"  Saved MC-dropout segmentation to {mc_segmentation_path}")

        print("\n" + "="*50)

        # image_files contains just the names (e.g., ['groceries.jpg', 'truck.jpg'])
        # output_dir is where the generated images are
        # image_dir is where the original assets are
        try:
            create_stitched_comparison(image_files, output_dir)
            print("Pipeline finished successfully!")
        except Exception as e:
            print(f"Error during stitching: {e}")
    print("All inferences complete! Generating 3x3 stitched comparisons...")
    
 

if __name__ == "__main__":
    main()