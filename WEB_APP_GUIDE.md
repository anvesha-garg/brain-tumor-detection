# 🧠 Brain Tumor Detection - Web App Guide

## 🚀 Quick Start

### Option 1: Run from Jupyter Notebook (Easiest)
1. Open `notebooks/02_training.ipynb`
2. Run cells in order:
   - Cell 1: Data setup (create `data_multi/`)
   - Cell 2: Load 4-class data
   - Cell 3: Train the model (10 epochs)
   - Cell 4: Launch Gradio web app
3. Gradio will open at: **http://localhost:7860**

### Option 2: Run Standalone Python App
```bash
# From project root directory
python app.py
```
Then open: **http://localhost:7860**

---

## 📋 Prerequisites

Make sure you have Gradio installed:
```bash
pip install gradio
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use the Web App

### Step 1: Upload an MRI Image
- Click "Upload MRI Scan" button
- Select a brain MRI image (JPG, PNG, BMP)
- Recommended size: 224×224 pixels or larger

### Step 2: Analyze
- Click the blue **"🔍 Analyze Image"** button
- Wait for prediction (usually < 1 second)

### Step 3: View Results
- **Left panel**: Original MRI image
- **Right panel**: Confidence scores for all 4 classes
  - 🔴 Glioma
  - 🔵 Meningioma
  - 🟢 Pituitary
  - 🟡 No Tumor

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | ResNet18 (pretrained on ImageNet) |
| **Classes** | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| **Input Size** | 224×224 pixels |
| **Device** | CPU or GPU (auto-detected) |

---

## 📂 Data Structure

The app expects data in this structure:
```
data_multi/
├── Training/
│   ├── glioma/           (33% of tumor images)
│   ├── meningioma/       (33% of tumor images)
│   ├── pituitary/        (33% of tumor images)
│   └── no_tumor/         (all no-tumor images, 80/20 split)
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

**Run Cell 1 of the notebook to auto-generate this structure!**

---

## 🔧 Configuration

Edit these in `app.py` to customize:

```python
# Change port number
interface.launch(server_port=8000)

# Enable sharing (generates public link)
interface.launch(share=True)

# Change to GPU
DEVICE = torch.device('cuda')
```

---

## 💾 Saving/Loading Models

### After Training
Models are automatically saved to:
```
models/brain_tumor_4class.pth
```

### Loading in Gradio App
The app automatically loads the best model from the `models/` folder.

### Manual Save
```python
torch.save(model.state_dict(), 'models/my_model.pth')
```

---

## ⚙️ Training Options

Modify these hyperparameters in Cell 3:

```python
num_epochs = 10              # Number of training epochs
learning_rate = 0.001        # Learning rate
batch_size = 16             # Batch size
```

---

## 🛠️ Troubleshooting

### Issue: "No data_multi/ folder found"
**Solution**: Run Cell 1 of the notebook to create the data structure

### Issue: "CUDA out of memory"
**Solution**: Use CPU or reduce batch size
```python
DEVICE = torch.device('cpu')
```

### Issue: Port 7860 already in use
**Solution**: Use a different port
```python
interface.launch(server_port=7861)
```

### Issue: Low accuracy
**Solution**: 
- Increase `num_epochs` to 20-30
- Check image quality in `data_multi/`
- Ensure consistent image formats

---

## 📝 Important Disclaimers

⚠️ **This is a demonstration system for educational purposes only!**

- ❌ NOT for medical diagnosis
- ❌ NOT a substitute for professional medical analysis
- ✅ Results are AI predictions only
- ✅ Always consult qualified medical professionals

---

## 🌐 Deploying to Cloud

### Hugging Face Spaces (Free)
1. Upload your notebook to Hugging Face
2. Create a Space with Gradio template
3. Push your code
4. Share public link

### Heroku / Railway (Paid)
```bash
# Create Procfile
web: python app.py

# Deploy
git push heroku main
```

---

## 📚 Next Steps

### Improve Model Accuracy
- Collect more data (1000+ images per class)
- Use data augmentation
- Hyperparameter tuning
- Ensemble multiple models

### Enhance Web App
- Add image preprocessing filters
- Show class probabilities over time
- Add batch prediction
- Save prediction history

### Add Advanced Features
- Compare with other models
- Explainability (GradCAM visualization)
- Batch processing
- Multi-image selection

---

## 🆘 Support

For issues with:
- **PyTorch**: https://pytorch.org/get-started
- **Gradio**: https://gradio.app/
- **ResNet**: https://arxiv.org/abs/1512.03385

---

## 📄 License

Educational use only. See LICENSE file.
