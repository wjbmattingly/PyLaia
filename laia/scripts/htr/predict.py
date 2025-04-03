import argparse
import json
from pathlib import Path
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.serialization import add_safe_globals

from laia.engine.engine_module import EngineModule
from laia.models.htr.laia_crnn import LaiaCRNN
from laia.losses.ctc_loss import CTCLoss
from laia.common.arguments import OptimizerArgs, SchedulerArgs

def load_char_map(char_map_path):
    """Load character map and create index to char mapping"""
    with open(char_map_path, 'r') as f:
        char_map = json.load(f)
    # Create reverse mapping (index -> char)
    idx_to_char = {v: k for k, v in char_map.items()}
    return idx_to_char, len(char_map)

def preprocess_image(image_path, data_dir=None):
    """Load and preprocess image for model input"""
    # Handle image path relative to data directory if provided
    if data_dir:
        # If image_path already starts with data_dir, don't add it again
        if str(image_path).startswith(str(data_dir)):
            image_path = Path(image_path)
        else:
            # Keep the full relative path (including train/ subdirectory)
            image_path = Path(data_dir) / 'images' / image_path
    else:
        image_path = Path(image_path)
    
    # Ensure path exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    # Load image in grayscale
    img = Image.open(image_path).convert('L')
    
    # Get image size
    w, h = img.size
    print(f"Original image size: {w}x{h}")
    
    # Normalize height to 128 pixels while maintaining aspect ratio
    target_height = 128
    new_width = int(w * (target_height / h))
    img = img.resize((new_width, target_height), Image.Resampling.BILINEAR)
    print(f"Resized image size: {new_width}x{target_height}")
    
    # Define transforms - match training normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Same as training
    ])
    
    # Apply transforms
    img_tensor = transform(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    print(f"Tensor value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"Tensor mean: {img_tensor.mean():.3f}")
    print(f"Tensor std: {img_tensor.std():.3f}")
    
    return img_tensor

def decode_predictions(output, idx_to_char):
    """Convert model output to text"""
    print(f"\nDebug information:")
    print(f"Output shape: {output.shape}")  # Should be [T, N, C]
    print(f"Number of classes: {len(idx_to_char)}")
    
    # Get predictions
    probs = torch.nn.functional.softmax(output, dim=2)  # [T, N, C]
    best_paths = torch.argmax(probs, dim=2)  # [T, N]
    
    print(f"Best paths shape: {best_paths.shape}")
    print(f"Sample predictions (first 10 timesteps): {best_paths[:10, 0].tolist()}")
    
    # Get top-3 predictions for first few timesteps
    top_probs, top_indices = torch.topk(probs[:10, 0, :], k=3, dim=1)
    print("\nTop-3 predictions for first 10 timesteps:")
    for t, (indices, probabilities) in enumerate(zip(top_indices, top_probs)):
        chars = [idx_to_char.get(idx.item(), '?') for idx in indices]
        probs_str = [f"{p.item():.4f}" for p in probabilities]
        print(f"t={t}: {list(zip(chars, probs_str))}")
    
    # Convert to text (simple greedy decoding)
    text = []
    prev_char = None
    for t in range(best_paths.size(0)):  # Iterate over time steps
        char_idx = best_paths[t, 0].item()  # Take first batch item
        if char_idx != 0 and char_idx != prev_char:  # Skip blank and repeated characters
            char = idx_to_char.get(char_idx, '?')
            text.append(char)
            prev_char = char_idx
    
    result = ''.join(text)
    print(f"\nFinal decoded text length: {len(result)}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--char_map', type=str, required=True,
                      help='Path to character map JSON file')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Path to data directory containing images')
    parser.add_argument('--invert_colors', action='store_true',
                      help='Invert image colors (black->white, white->black)')
    args = parser.parse_args()

    # Load character map
    idx_to_char, num_classes = load_char_map(args.char_map)
    print(f"\nLoaded character map with {num_classes} classes")
    print("Sample characters:", {i: c for i, c in list(idx_to_char.items())[:10]})
    
    # Initialize model with parameters matching the checkpoint
    cnn_features = [16, 32, 48, 64]  # CNN feature dimensions
    model = LaiaCRNN(
        num_input_channels=1,  # Grayscale images
        num_output_labels=num_classes,
        cnn_num_features=cnn_features,  # Number of features per CNN layer (4 layers)
        cnn_kernel_size=[(3, 3)] * 4,  # 3x3 kernels for all layers
        cnn_stride=[(1, 1)] * 4,  # No stride
        cnn_dilation=[(1, 1)] * 4,  # No dilation
        cnn_activation=[torch.nn.ReLU] * 4,  # ReLU activation
        cnn_poolsize=[(2, 2)] * 4,  # 2x2 pooling
        cnn_dropout=[0.0] * 4,  # No dropout
        cnn_batchnorm=[True] * 4,  # Use batch normalization
        image_sequencer="maxpool-8",  # Use maxpool-8 to get 512 input features (64 * 8 = 512)
        rnn_units=256,  # Matches checkpoint (1024/4 since bidirectional=True and we have 2 directions)
        rnn_layers=3,
        rnn_dropout=0.5,
        lin_dropout=0.5,
    )
    print("\nModel configuration:")
    print(f"CNN features: {cnn_features}")
    print(f"Last CNN layer output features: {cnn_features[-1]}")
    print(f"Image sequencer: maxpool-8 ({cnn_features[-1]} * 8 = {cnn_features[-1] * 8} RNN input features)")
    print(f"RNN units: {256 * 2} (bidirectional)")
    print(f"Number of classes: {num_classes}")
    criterion = CTCLoss()
    
    # Add all necessary safe globals
    add_safe_globals([
        OptimizerArgs, 
        SchedulerArgs,
        getattr,  # Required for loading Lightning checkpoints
        LaiaCRNN,  # Required for model class
        CTCLoss,   # Required for criterion class
        EngineModule  # Required for engine module
    ])
    
    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print("Checkpoint loaded successfully")
    
    # Extract model state dict from checkpoint
    if 'state_dict' in checkpoint:
        # Handle Lightning checkpoint
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                # Remove 'model.' prefix
                state_dict[key[6:]] = value
            else:
                state_dict[key] = value
    else:
        # Handle direct model state dict
        state_dict = checkpoint
    
    print("\nApplying weights to model...")
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully and moved to device: {device}")
    
    # Preprocess image
    print(f"\nProcessing image: {args.image}")
    img_tensor = preprocess_image(args.image, args.data_dir)
    print(f"Input tensor shape: {img_tensor.shape}")
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)  # Use model directly
        print(f"Raw model output shape: {output.shape}")
    
    # Decode predictions
    text = decode_predictions(output, idx_to_char)
    print(f"\nPredicted text: {text}")

if __name__ == '__main__':
    main() 