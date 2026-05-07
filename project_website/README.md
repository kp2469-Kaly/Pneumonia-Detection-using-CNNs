# Project Website

This folder contains the GitHub Pages website for the Pneumonia Detection project. It is a fully static multi-page site — no server or build step required.

## Files

| File | Purpose |
|---|---|
| `index.html` | Home page |
| `methodology.html` | Dataset, preprocessing, architecture, and training pipeline |
| `results.html` | Metrics tables/charts and visual prediction outputs |
| `demo.html` | Live in-browser ONNX inference page |
| `style.css` | Custom CSS (blue, orange, white theme) |
| `script.js` | Scroll animations, AUC bar chart, and ONNX inference logic |
| `densenet121_best.onnx` | ONNX model for in-browser inference (generate with `export_to_onnx.py`) |
| `densenet121_best.onnx.data` | External ONNX tensor data file used by the model |
| `assets/normal.png` | Sample prediction — NORMAL class |
| `assets/pneumonia_true.png` | Sample prediction — PNEUMONIA class |

## Live Demo Setup (required before deploying)

The **"Try It Live"** section runs the real DenseNet121 model inside the visitor's browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). The model file is not stored in git by default because it is ~30 MB — you need to generate it once and commit it.

From the project root, run:

```bash
py export_to_onnx.py
```

This creates `project_website/densenet121_best.onnx` and `project_website/densenet121_best.onnx.data`.
Commit both files before pushing.

## Deploying to GitHub Pages

1. Run `export_to_onnx.py` and commit both generated model files
2. Open your repository on GitHub → **Settings → Pages**
3. Under **Build and deployment**, choose:
  - Source: **Deploy from a branch**
  - Branch: `main`
  - Folder: `/` (root) if this folder is the selected Pages source, or `/docs` if you copy website files into `docs/`
4. Click **Save** and wait about a minute

## Repository Link

In `script.js`, set `repoUrl` to the repository URL so the GitHub button points to the project page:

```js
const repoUrl = "https://github.com/<username>/<repository>";
```

## Local Testing

Do **not** open `index.html` directly from the file system — browsers block loading `.onnx` files over the `file://` protocol. Use a local HTTP server instead:

```bash
cd project_website
python -m http.server 8080
# then open http://localhost:8080 in your browser
```
