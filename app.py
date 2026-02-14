"""
🧠 Brain Tumor Detection - Gradio Web App
Priority 3: Complete web interface for 4-class classification
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import io
import base64

# ================================================================================
# CONFIGURATION
# ================================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_MULTI = PROJECT_ROOT / "data_multi"
MODELS_DIR = PROJECT_ROOT / "models"

# Class names (must match your data_multi/ structure)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"✅ Running on device: {DEVICE}")

# ================================================================================
# MODEL SETUP
# ================================================================================

def load_model(model_path=None):
    """Load the trained ResNet18 model."""
    if model_path is None:
        # Try to find the best model
        model_files = list(MODELS_DIR.glob("brain_tumor*.pth"))
        if not model_files:
            print("⚠️  No saved models found. Using untrained ResNet18.")
            model_path = None
        else:
            model_path = model_files[0]
    
    # Create model
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Load weights if exists
    if model_path and Path(model_path).exists():
        print(f"✅ Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle both direct state_dict and checkpoint dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully!")
    else:
        print("⚠️  No checkpoint found - using ImageNet pretrained weights only")
    
    model.eval()
    return model

# Load model
model = load_model()

# ================================================================================
# TRANSFORMS
# ================================================================================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================================================================================
# PREDICTION FUNCTION
# ================================================================================

def predict_tumor(image_input):
    """
    Predict brain tumor class from MRI image.
    
    Args:
        image_input: PIL Image or numpy array
    
    Returns:
        tuple: (predictions_dict, visualization_image)
    """
    
    if image_input is None:
        return {"error": "Please upload an image"}, None
    
    try:
        # Convert to PIL if needed
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        # Store original for display
        original_image = image.copy()
        
        # Transform
        image_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[predicted_class].item()
        
        # Create results dictionary
        results = {
            CLASS_NAMES[i]: float(probabilities[i].cpu().numpy())
            for i in range(NUM_CLASSES)
        }
        
        # Create visualization
        viz_image = create_visualization(
            original_image, 
            CLASS_NAMES[predicted_class], 
            results,
            confidence
        )
        
        return results, viz_image
    
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg}, None

# ================================================================================
# VISUALIZATION
# ================================================================================

def create_visualization(image, predicted_class, probabilities, confidence):
    """Create a visualization with image + results."""
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title("Input MRI Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Predictions chart
    classes = list(probabilities.keys())
    scores = list(probabilities.values())
    colors = ['#ff6b6b' if c == predicted_class else '#4ecdc4' for c in classes]
    
    axes[1].barh(classes, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    
    # Add value labels
    for i, (cls, score) in enumerate(zip(classes, scores)):
        axes[1].text(score + 0.02, i, f'{score:.1%}', va='center', fontweight='bold')
    
    # Highlight predicted class
    axes[1].text(0.5, -0.15, 
                f"🎯 Predicted: {predicted_class.upper()} ({confidence:.1%})", 
                transform=axes[1].transAxes,
                fontsize=13, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close(fig)
    
    return result_image

# ================================================================================
# GRADIO INTERFACE
# ================================================================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="🧠 Brain Tumor Detection", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🧠 Brain Tumor Detection System
        
        **4-Class MRI Classification**: Glioma | Meningioma | Pituitary | No Tumor
        
        ---
        
        Upload an MRI brain scan image to get predictions with confidence scores.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload MRI Image")
                image_input = gr.Image(
                    label="Brain MRI Scan",
                    type="pil",
                    sources=["upload", "webcam"],
                    interactive=True
                )
                
                submit_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                predictions_output = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=4
                )
        
        # Visualization
        gr.Markdown("### 📈 Detailed Visualization")
        viz_output = gr.Image(
            label="Prediction Chart & Input Image",
            type="pil"
        )
        
        # Model info
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"""
                ### ℹ️ Model Information
                
                - **Architecture**: ResNet18 (ImageNet pretrained)
                - **Classes**: {NUM_CLASSES} (Glioma, Meningioma, Pituitary, No Tumor)
                - **Device**: {DEVICE}
                - **Input Size**: 224×224 pixels
                
                **How to use**:
                1. Upload a brain MRI image
                2. Click "Analyze Image"
                3. View predictions and confidence scores
                """)
            
            with gr.Column():
                gr.Markdown("""
                ### 📋 Important Notes
                
                ⚠️ **Disclaimer**: This is a demonstration system for educational purposes.
                
                - Always consult medical professionals for diagnosis
                - Results are AI predictions, not medical advice
                - Different imaging protocols may affect results
                - Quality of input image impacts accuracy
                
                ### Supported Formats
                - JPG, PNG, BMP, GIF
                - Recommended: 224×224 or larger
                - Grayscale or color images
                """)
        
        # Event handler
        submit_btn.click(
            fn=predict_tumor,
            inputs=image_input,
            outputs=[predictions_output, viz_output]
        )
        
        # Example images (if you have any)
        example_paths = list(Path(DATA_MULTI / "Testing").glob("*/**.jpg"))[:4]
        if example_paths:
            gr.Examples(
                examples=[[str(p)] for p in example_paths],
                inputs=image_input,
                outputs=[predictions_output, viz_output],
                fn=predict_tumor,
                label="Example MRI Images",
                cache_examples=False
            )
    
    return demo

# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Brain Tumor Detection Web App")
    print("=" * 60)
    
    # Check data
    if not DATA_MULTI.exists():
        print(f"⚠️  Warning: {DATA_MULTI} not found")
        print("Make sure to run the data setup cell first!")
    else:
        print(f"✅ Data path: {DATA_MULTI}")
    
    # Create interface
    interface = create_interface()
    
    # Launch
    print("\n🌐 Starting Gradio server...")
    print("📱 Open this URL in your browser: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
