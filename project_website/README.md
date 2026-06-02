# Project Website

This folder contains the GitHub Pages website for the Pneumonia Detection project. It is a fully static multi-page site - no server or build step required.

The Try Model page includes a browser-side upload-and-predict feature powered by ONNX Runtime Web.

## Files

| File | Purpose |
|---|---|
| `index.html` | Home page |
| `methodology.html` | Dataset, preprocessing, architecture, and training pipeline |
| `results.html` | Metrics tables/charts and visual prediction outputs |
| `try-model.html` | Upload image and run in-browser inference |
| `style.css` | Custom CSS (blue, orange, white theme) |
| `script.js` | Scroll animations, dynamic metrics rendering, and ONNX inference logic |
| `assets/densenet121_best.onnx` | Exported web model used for in-browser inference |
| `assets/normal.png` | Sample prediction - NORMAL class |
| `assets/pneumonia_true.png` | Sample prediction - PNEUMONIA class |

## Enable Upload Inference

Before deploying the site, export the PyTorch checkpoint to ONNX:

```bash
python model_artifacts/export_to_onnx.py \
  --checkpoint model_artifacts/densenet121_best.pth \
  --output project_website/assets/densenet121_best.onnx
```

If the ONNX file is missing, the upload UI will show an inference error.

## Deploying to GitHub Pages

1. Open your repository on GitHub → **Settings → Pages**
2. Under **Build and deployment**, choose:
  - Source: **Deploy from a branch**
  - Branch: `main`
  - Folder: `/` (root) if this folder is the selected Pages source, or `/docs` if you copy website files into `docs/`
3. Click **Save** and wait about a minute

## Repository Link

In `script.js`, set `repoUrl` to the repository URL so the GitHub button points to the project page:

```js
const repoUrl = "https://github.com/<username>/<repository>";
```

## Local Testing

Use a local HTTP server for testing:

```bash
cd project_website
python -m http.server 8080
# then open http://localhost:8080 in your browser
```

Open `http://localhost:8080/try-model.html`, upload a chest X-ray, and click **Predict**.

> This model is trained for chest X-rays only (NORMAL vs PNEUMONIA). It is not valid for kidney scans or other modalities.

