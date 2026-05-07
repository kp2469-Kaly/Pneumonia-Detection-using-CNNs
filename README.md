# Pneumonia Detection from Chest X-rays

A deep learning project for pneumonia detection in chest X-rays. The goal is to classify pediatric chest X-rays as either **NORMAL** or **PNEUMONIA** using a progression of three model architectures, with a focus on practical clinical considerations like class imbalance, conservative augmentation, and explainability through Grad-CAM.

---

## Results

Three models were trained and evaluated on the same held-out test set (624 images). All numbers below are real - pulled directly from notebook cell outputs.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 0.8029 | 0.7846 | 0.9436 | 0.8568 | 0.8995 |
| ResNet-18 | 0.8045 | 0.7627 | 0.9974 | 0.8644 | 0.9438 |
| **DenseNet121** | **0.8526** | **0.8104** | **0.9974** | **0.8943** | **0.9557** |

DenseNet121 was selected as the final model. The near-perfect recall (0.9974) is intentional - in a clinical screening context, missing a pneumonia case is far more costly than a false alarm.

---

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)** by Paul Mooney  
Available at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains anterior-posterior chest radiographs from pediatric patients, organized into `NORMAL` and `PNEUMONIA` folders. We used the official `train` and `test` splits, with a stratified validation split carved out of training data.

| Split | Images |
|---|---|
| Train | 4,433 |
| Validation | 783 |
| Test | 624 |

> **Note:** The dataset is not included in this repository. The notebook uses `kagglehub` to download it automatically at runtime. You will need a Kaggle account and API credentials configured.

---

## Project Structure

```
project/
├── pneumonia_detection_notebook.ipynb   # Full end-to-end pipeline
├── export_to_onnx.py                    # Exports checkpoint to ONNX for the web demo
├── requirements.txt                     # Python dependencies (pip)
├── environment.yml                      # Conda environment (recommended)
├── model_artifacts/
│   ├── densenet121_best.pth             # Saved best model checkpoint
│   ├── normal.png                       # Sample prediction - NORMAL
│   ├── pneumonia_true.png               # Sample prediction - PNEUMONIA
│   └── use_pneumonia_model.py           # CLI inference script
└── project_website/
    ├── index.html                       # GitHub Pages website
    ├── style.css
    ├── script.js                        # Includes ONNX Runtime Web inference
    ├── densenet121_best.onnx            # ONNX model for in-browser inference (generate with export_to_onnx.py)
    └── assets/
        ├── normal.png
        └── pneumonia_true.png
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
```

### 2. Set up the environment

**Option A - Conda (recommended, matches the training environment):**

```bash
conda env create -f environment.yml
conda activate pneumonia-detection
```

**Option B - pip:**

```bash
pip install -r requirements.txt
```

> PyTorch with CUDA: if you have a GPU, install the CUDA-enabled build of PyTorch from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`. The command will vary depending on your CUDA version.

### 3. Configure Kaggle credentials

The notebook downloads the dataset automatically via `kagglehub`. You need a Kaggle API key:

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. This downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)

### 4. Run the notebook

```bash
jupyter notebook pneumonia_detection_notebook.ipynb
```

Run all cells from top to bottom. Training runs automatically and saves the best checkpoint to `model_artifacts/densenet121_best.pth`.

---

## Inference on a Single Image

After training (or if you already have the checkpoint), you can classify any chest X-ray with one command:

```bash
python model_artifacts/use_pneumonia_model.py path/to/xray.jpg
```

Example output:
```
Using device: cpu
Loaded DenseNet121 (test ROC-AUC during training: 0.9557)

Image:      xray.jpg
Prediction: PNEUMONIA
  P(PNEUMONIA) = 0.9832
  P(NORMAL)    = 0.0168
```

You can also point to a specific checkpoint:

```bash
python model_artifacts/use_pneumonia_model.py path/to/xray.jpg --checkpoint model_artifacts/densenet121_best.pth
```

---

## Live Web Demo

The project website includes a **"Try It Live"** section where anyone can upload a chest X-ray and get a prediction directly in the browser - no server involved. It uses [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) to run the DenseNet121 model via WebAssembly.

To enable it, first export the model to ONNX format:

```bash
py export_to_onnx.py
```

This creates two files used by ONNX Runtime Web:

- `project_website/densenet121_best.onnx`
- `project_website/densenet121_best.onnx.data`

Commit both files so the browser demo can run correctly.

---

## Model Details

### Architecture

All three models output a single sigmoid probability: P(PNEUMONIA). The decision threshold is 0.5.

- **Baseline CNN**: Three conv-pool blocks followed by a fully connected classifier. Trained from scratch.
- **ResNet-18**: ImageNet-pretrained backbone with the final fully connected layer replaced by `Linear(512, 1) → Sigmoid`.
- **DenseNet121**: ImageNet-pretrained backbone with the classifier replaced by `Linear(1024, 1) → Sigmoid`. Best overall performance.

### Training Configuration

| Setting | Baseline CNN | ResNet-18 | DenseNet121 |
|---|---|---|---|
| Learning rate | 1e-3 | 1e-4 | 1e-4 |
| Batch size | 32 | 32 | 32 |
| Max epochs | 10 | 10 | 10 |
| Early stopping patience | 3 | 3 | 3 |
| Loss | BCELoss | BCELoss | BCELoss |
| Optimizer | Adam (wd=1e-4) | Adam (wd=1e-4) | Adam (wd=1e-4) |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |

### Preprocessing & Augmentation

Images are resized to 224�-224 and converted from grayscale to 3 channels (to be compatible with ImageNet-pretrained weights). Augmentation is kept conservative to avoid introducing unrealistic distortions into medical images:

- Random horizontal flip (p=0.3)
- Random rotation (±7°)
- Normalized with ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

Class imbalance in the training set is handled using `WeightedRandomSampler`.

### Explainability

The notebook includes **Grad-CAM** visualizations for the DenseNet121 model. These overlay a heatmap on the original X-ray to highlight which regions of the image most influenced the prediction, making the model's reasoning more interpretable.

---

## Deploying the Website to GitHub Pages

1. Run `py export_to_onnx.py` and commit both model files:
  - `project_website/densenet121_best.onnx`
  - `project_website/densenet121_best.onnx.data`
2. Push the full repository to GitHub.
3. For GitHub Pages, publish from `/docs` or repository root (`/`) on branch `main`.
4. If using `/docs`, copy `project_website/*` into `docs/` before deployment.
5. Open **Settings → Pages** and select the matching source/folder.
6. Update `repoUrl` in `project_website/script.js` to the repository URL.

---

## Disclaimer

This project is a research and academic exercise. The model is not validated for clinical use and should not be used to make or influence any medical decisions.


