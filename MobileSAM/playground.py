from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import torch
model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)

image = cv2.imread('video01_000001_1.png')
cv2.imshow('  in put Image', image.astype((np.uint8)))
cv2.waitKey(1)
predictor.set_image(image)
input_point = np.array([[200, 158],[170,123]])
input_label = np.array([1,2])
masks,  scores, logits = predictor.predict( point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,)
print(masks.shape)
for i, (mask, score) in enumerate(zip(masks, scores)):
    alpha= 0.5
    # overlay = cv2.addWeighted(mask.astype((np.uint8)), 1 - alpha, image.astype((np.uint8)), alpha, 0)
    overlay = image * mask[:, :, np.newaxis]
    cv2.imshow(f"Mask {i+1}, Score: {score:.3f}", overlay.astype((np.uint8)))
    cv2.waitKey(1)