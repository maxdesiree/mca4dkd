IMCA4DKD: Multimodal Cross-Attention for Diabetic Kidney Disease Detection

This repository contains the official implementation of IMCA4DKD, a lightweight multimodal deep learning framework that integrates 12-lead ECG images and routine clinical vital signs using bidirectional cross-attention. The model is designed for early risk detection of diabetic kidney disease (DKD), particularly in resource-limited healthcare settings where laboratory testing or imaging may be inaccessible.

🔍 IMCA4DKD performs feature-level fusion between two data streams:
ECG images → encoded using ResNet50 with channel attention
Clinical tabular features (pulse, SBP, DBP, height, weight, sex, age) → encoded with self-attention
Bidirectional cross-attention aligns ECG and clinical representations
Fusion + ensemble heads produce the final prediction

We additionally provide an explainability module (XAI) able to quantify:
ECG heatmaps (Grad-CAM)
How does ECG change the importance of each clinical feature

Contact for usage: Prof Dinesh Kumar at dinesh.kumar@rmit.edu
