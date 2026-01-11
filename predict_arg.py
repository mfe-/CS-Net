
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
from model.csnet import CSNet
import torch.serialization
from skimage.filters import threshold_otsu

def parse_args():
    parser = argparse.ArgumentParser(description="CS-Net Single Image Prediction")
    parser.add_argument("model_path", type=str, nargs='?', default=None, help="Path to trained model checkpoint (.pkl). If not provided, the latest in 'checkpoint/' will be used.")
    parser.add_argument("image_path", type=str, help="Path to input image (png/tif)")
    parser.add_argument("output_path", type=str, help="Base path for output mask (extension will be added)")
    parser.add_argument("--img-size", type=int, default=None, help="Resize input image to this size (default: None, no resizing)")
    args = parser.parse_args()

    # Handle default model_path logic here
    if args.model_path is None:
        ckpt_dir = "checkpoint"
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError("No model_path provided and 'checkpoint/' directory does not exist.")
        import re
        ckpt_files = [f for f in os.listdir(ckpt_dir) if re.match(r'CS_Net_DRIVE_\d+\.pkl', f)]
        if not ckpt_files:
            raise FileNotFoundError("No checkpoint files found in 'checkpoint/' directory.")
        # Extract numbers and find the highest
        max_ckpt = max(ckpt_files, key=lambda x: int(re.findall(r'CS_Net_DRIVE_(\d+)\.pkl', x)[0]))
        args.model_path = os.path.join(ckpt_dir, max_ckpt)
        print(f"Using latest checkpoint: {args.model_path}")
    return args

def predict_single_image(model_path, image_path, output_path, img_size=None):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.serialization.add_safe_globals([torch.nn.DataParallel])
    net = torch.load(model_path, map_location=device, weights_only=False)
    net.eval()

    # Prepare image
    # If the input image is a PNG, convert to TIFF first
    if image_path.lower().endswith('.png'):
        temp_tif_path = os.path.splitext(image_path)[0] + '_temp.tif'
        img_png = Image.open(image_path).convert('RGB')
        img_png.save(temp_tif_path, format='TIFF')
        image_path = temp_tif_path
    image = Image.open(image_path).convert('RGB')
    if img_size is not None:
        image = image.resize((img_size, img_size))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = net(image)
        # Probability map (0-255 grayscale)
        prob = output.sigmoid().cpu().numpy()[0, 0]
        prob_map = prob * 255
        prob_map = prob_map.astype(np.uint8)
        # Print statistics for debugging
        print(f"Probability map stats: min={prob.min():.4f}, max={prob.max():.4f}, mean={prob.mean():.4f}")
        # Automatically calculate threshold using Otsu's method

        otsu_thresh = threshold_otsu(prob)
        print(f"Otsu's threshold: {otsu_thresh:.4f}")
        bitmask = (prob > otsu_thresh) * 255
        bitmask = bitmask.astype(np.uint8)

        # Save probability map as PNG
        prob_map_path = os.path.splitext(output_path)[0] + '_prob.png'
        Image.fromarray(prob_map).save(prob_map_path, format='PNG')
        print(f"Probability map saved to {prob_map_path}")

        # Save bitmask as GIF
        bitmask_path = os.path.splitext(output_path)[0] + '_bitmask.gif'
        Image.fromarray(bitmask).save(bitmask_path, format='GIF')
        print(f"Bitmask saved to {bitmask_path}")

if __name__ == '__main__':
    args = parse_args()
    predict_single_image(args.model_path, args.image_path, args.output_path, img_size=args.img_size)
