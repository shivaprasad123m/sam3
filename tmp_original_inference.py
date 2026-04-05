import os
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.dirname(os.path.abspath(__file__))
bpe_path = os.path.join(sam3_root, 'sam3', 'assets', 'bpe_simple_vocab_16e6.txt.gz')
log_path = os.path.join(sam3_root, 'tmp_original_inference.log')

with open(log_path, 'w', encoding='utf-8') as log:
    model = build_sam3_image_model(bpe_path=bpe_path, device='cpu')
    model = model.to(dtype=torch.float32)
    log.write('model loaded\n')
    
    # Test both images
    for image_name, prompt in [('truck.jpg', 'truck'), ('groceries.jpg', 'bags')]:
        image_path = os.path.join(sam3_root, 'assets', 'uncertainImages', image_name)
        log.write(f'\n=== {image_name} with prompt "{prompt}" ===\n')
        log.write(f'image_path={image_path}\n')
        log.write(f'exists={os.path.exists(image_path)}\n')
        
        processor = Sam3Processor(model, confidence_threshold=0.1, device='cpu')
        image = Image.open(image_path)
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        
        scores = state.get('scores')
        boxes = state.get('boxes')
        masks_logits = state.get('masks_logits')
        masks = state.get('masks')
        
        log.write(f'scores={scores}\n')
        log.write(f'boxes={None if boxes is None else boxes.shape}\n')
        log.write(f'masks_logits={None if masks_logits is None else masks_logits.shape}\n')
        log.write(f'masks_sum={None if masks is None else masks.sum().item()}\n')
        
        if boxes is not None and len(boxes) > 0:
            log.write('First few boxes:\n')
            for i in range(min(5, len(boxes))):
                box = boxes[i].cpu().numpy()
                score = scores[i].item() if scores is not None else 0
                log.write(f'  Box {i}: {box} score={score:.3f}\n')
