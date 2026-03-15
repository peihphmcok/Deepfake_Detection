# Deepfake Detection

Multimodal deepfake detection (face + voice) for research and reproduction. This repository contains training and evaluation code for face-based, audio-based, and decision-level fusion models.

## Repository structure

```
Deepfake_Detection/
├── classification/
│   ├── face/          # Face-based detection (Xception, baseline)
│   ├── voice/         # Audio-based detection (CRNN, baseline)
│   └── multimodal/    # Fusion & evaluation
├── data_preprocessing/ # Data extraction, labeling, splitting
├── data/              # Dataset download scripts (FaceForensics++, etc.)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- CUDA (optional, for GPU training)

## Installation

```bash
git clone https://github.com/peihphmcok/Deepfake_Detection.git
cd Deepfake_Detection
pip install -r requirements.txt
```

Install PyTorch with CUDA if needed (see [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version):

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data

- **Face:** Use scripts in `data/FaceForensics_c23/` and `data/FaceForensics_c40/` to download FaceForensics++.
- **Audio/Video:** Preprocessing scripts in `data_preprocessing/` support custom datasets (e.g. FakeAVCeleb-style).

## Usage

- **Face training:** e.g. `python -m classification.face.Implementation.baseline.baseline_train` (adjust paths in script).
- **Voice training:** run scripts in `classification/voice/Implementation/` (baseline or CRNN).
- **Multimodal fusion:** use `classification/multimodal/decision_level_fusion.py` and related evaluation scripts.

## Citation

If you use this code in your research, please cite the paper (update with your BibTeX when published):

```bibtex
@article{yourpaper2025,
  title={...},
  author={...},
  year={2025}
}
```

## License

See repository license file.
