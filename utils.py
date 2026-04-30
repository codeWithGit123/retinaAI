import torch
import numpy as np
import matplotlib.cm as cm


def generate_gradcam(model, image_tensor, target_layer, target_class):
    """
    Generate Grad-CAM heatmap for visualization of model decisions

    Args:
        model: PyTorch model
        image_tensor: Input image tensor (3, H, W)
        target_layer: Target layer for visualization
        target_class: Target class index

    Returns:
        grayscale_cam: Normalized CAM heatmap
    """
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = image_tensor.unsqueeze(0)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)

    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    fh.remove()
    bh.remove()

    grads = gradients[0]          # (1, C, H, W)
    acts  = activations[0]        # (1, C, H, W)

    # Global average pool the gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

    # Weighted sum of activations
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()                # (H, W)

    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=0.55):
    """
    Overlay a Grad-CAM heatmap on an image using matplotlib colormap.
    Pure NumPy/matplotlib — no opencv required.

    Args:
        img: float32 numpy array (H, W, 3) in range [0, 1]
        grayscale_cam: float32 numpy array (H, W) in range [0, 1]
        use_rgb: if True output is RGB, else BGR
        image_weight: blend weight for original image (0-1)

    Returns:
        visualization: uint8 numpy array (H, W, 3)
    """
    # Resize CAM to match image if needed
    if grayscale_cam.shape != img.shape[:2]:
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray((grayscale_cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((img.shape[1], img.shape[0]), PILImage.BILINEAR)
        grayscale_cam = np.array(cam_pil).astype(np.float32) / 255.0

    # Apply colormap (jet)
    colormap = cm.get_cmap('jet')
    heatmap = colormap(grayscale_cam)[:, :, :3].astype(np.float32)  # (H, W, 3), drop alpha

    # Blend
    cam_weight = 1.0 - image_weight
    visualization = image_weight * img + cam_weight * heatmap
    visualization = np.clip(visualization, 0, 1)
    visualization = (visualization * 255).astype(np.uint8)

    return visualization


def process_image_for_gradcam(image_array):
    """
    Convert image to normalised float format for Grad-CAM overlay.

    Args:
        image_array: PIL Image or numpy array

    Returns:
        Normalised numpy array (float32, values in [0, 1])
    """
    if isinstance(image_array, np.ndarray):
        img = image_array.astype(np.float32) / 255.0
    else:
        img = np.array(image_array).astype(np.float32) / 255.0

    return img