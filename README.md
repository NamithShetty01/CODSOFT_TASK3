# 🖼️ CODSOFT Task 3 - Image Captioning AI

An AI-powered image captioning system developed in Python as part of the **CodSoft Artificial Intelligence Internship**.

The system analyzes an input image using a pre-trained **Vision Transformer (ViT)** and **GPT-2** model and automatically generates a natural-language description of the image.

---

## 📌 Project Description

Image Captioning is an AI task that combines **Computer Vision** and **Natural Language Processing (NLP)**.

This project uses the pre-trained:

- Vision Transformer (ViT) for image understanding
- GPT-2 for generating natural-language captions
- Hugging Face Transformers for model implementation
- PIL for image processing
- PyTorch for deep learning operations

The user provides an image path, and the system analyzes the image and generates a descriptive caption.

---

## ✨ Features

- 🖼️ Accepts JPG, PNG and other supported image formats
- 🤖 Uses a pre-trained image captioning model
- 👁️ Analyzes visual content using Vision Transformer
- 📝 Generates natural-language captions using GPT-2
- 🔄 Allows multiple images to be tested
- ❌ Handles invalid image paths
- 🚪 Supports `exit`, `quit`, and `bye` commands
- 💻 Runs directly from the command line

---

## 🧠 Model Architecture

```text
Input Image
     ↓
Image Preprocessing
     ↓
Vision Transformer (ViT)
     ↓
Visual Feature Extraction
     ↓
GPT-2 Language Model
     ↓
Text Generation
     ↓
Generated Image Caption