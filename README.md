# OSL Sign Language Recognition

## Overview
This project presents an Orange Sign Language (OSL) recognition system developed as a Final Year Project. The system translates sign language gestures into textual output using a deep learning pipeline.

The model combines Spatial-Temporal Graph Convolutional Networks (ST-GCN) for pose-based feature extraction with a sequence-to-sequence transformer model (mT5) for translation.

## Features
- Sign language recognition from pose sequences
- Word-level and sentence-level translation
- Real-time inference using webcam input
- Training and evaluation pipeline
- External evaluation metrics (BLEU, ROUGE)

## Model Architecture
The system consists of two main components:

1. ST-GCN:
   Extracts spatial-temporal features from human pose keypoints.

2. mT5:
   Translates extracted features into natural language sentences.

## Project Structure

├── OSL_Run_Pipeline/ # Training and evaluation pipeline
├── demo/ # Real-time inference and pose extraction
├── external_metrics/ # Evaluation metrics (BLEU, ROUGE)
├── stgcn_layers/ # ST-GCN model components
├── data/
│ └── labels/ # Label files (download separately)
├── checkpoints/ # Model weights (not included)
├── notebooks/ # Analysis and experiments
├── config.py
├── models.py
├── datasets.py
└── README.md

## Dataset
Due to size limitations, the dataset and trained model checkpoints are not included in this repository.
Label files can be downloaded from:
[will be relased later]

After downloading, place them in:
data/labels/

## Installation
Clone the repository:

git clone https://github.com/moha21114j/osl-sign-language-recognition.git
cd osl-sign-language-recognition

Install dependencies:

pip install -r requirements.txt

## Training
To train the model, run:

python OSL_Run_Pipeline/run_osl_training.py

## Inference
To run real-time sign recognition:

python demo_webcam.py

## Results
The system was evaluated using standard NLP metrics including BLEU and ROUGE scores.
## Acknowledgement
This project is built upon the Uni-Sign framework, “Toward Unified Sign Language Understanding at Scale.” The original implementation and research provided a strong foundation for developing the Orange Sign Language (OSL) recognition system presented in this work.

We acknowledge and appreciate the contributions of the authors of Uni-Sign for making their codebase and methodology publicly available. Their work on integrating Spatial-Temporal Graph Convolutional Networks (ST-GCN) with transformer-based models significantly influenced the design and implementation of this project.

This project adapts and extends the Uni-Sign framework to support Orange Sign Language, including dataset preparation, model training, and real-time inference components.
## Author
FYP Team
Final Year Project – Sultan Qaboos University
