# Project Website

This folder contains the GitHub Pages website for the Pneumonia Detection project. It is a fully static multi-page site - no server or build step required.

## Files

| File | Purpose |
|---|---|
| `index.html` | Home page |
| `methodology.html` | Dataset, preprocessing, architecture, and training pipeline |
| `results.html` | Metrics tables/charts and visual prediction outputs |
| `style.css` | Custom CSS (blue, orange, white theme) |
| `script.js` | Scroll animations and dynamic metrics rendering |
| `assets/normal.png` | Sample prediction - NORMAL class |
| `assets/pneumonia_true.png` | Sample prediction - PNEUMONIA class |

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

