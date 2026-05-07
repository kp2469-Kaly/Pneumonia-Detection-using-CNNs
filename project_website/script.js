const repoUrl = "";

const techStack = [
  "Python",
  "PyTorch",
  "Torchvision",
  "NumPy",
  "Pandas",
  "Matplotlib",
  "Scikit-learn",
  "KaggleHub"
];

const aucData = [
  { model: "Baseline", value: 0.8995 },
  { model: "ResNet18", value: 0.9438 },
  { model: "DenseNet121", value: 0.9557 }
];

function setRepoLinks() {
  const githubButton = document.getElementById("githubLink");
  const footerGithubLink = document.getElementById("footerGithubLink");

  if (repoUrl && repoUrl !== "#") {
    githubButton.href = repoUrl;
    footerGithubLink.href = repoUrl;
  }
}

function renderTech() {
  const grid = document.getElementById("techGrid");
  if (!grid) return;

  techStack.forEach((item) => {
    const pill = document.createElement("div");
    pill.className = "tech-pill";
    pill.textContent = item;
    grid.appendChild(pill);
  });
}

function renderAucBars() {
  const wrapper = document.getElementById("aucBars");
  if (!wrapper) return;

  aucData.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.textContent = entry.model;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, entry.value * 100))}%`;

    const value = document.createElement("strong");
    value.textContent = entry.value.toFixed(4);

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    wrapper.appendChild(row);
  });
}

function enableReveal() {
  const revealNodes = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  revealNodes.forEach((node) => observer.observe(node));
}

window.addEventListener("DOMContentLoaded", () => {
  setRepoLinks();
  renderTech();
  renderAucBars();
  enableReveal();
});

// ============================================================
// Live Demo — ONNX Runtime Web inference
// ============================================================

const ONNX_MODEL_URL = "densenet121_best.onnx";
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

let ortSession = null;
let ortLoading = false;

function setModelStatus(msg, showSpinner) {
  const text    = document.getElementById("modelStatusText");
  const spinner = document.getElementById("modelSpinner");
  if (text)    text.textContent = msg;
  if (spinner) spinner.style.display = showSpinner ? "inline-block" : "none";
}

async function ensureModel() {
  if (ortSession) return;
  if (ortLoading) throw new Error("Model is already loading.");
  ortLoading = true;

  setModelStatus("Loading DenseNet121 model (~30 MB) — please wait…", true);
  try {
    // Point ONNX Runtime WASM binaries to the same CDN version
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    ortSession = await ort.InferenceSession.create(ONNX_MODEL_URL);
    setModelStatus("Model loaded. Ready to analyze.", false);
  } catch (e) {
    ortLoading = false;
    setModelStatus("Failed to load model: " + e.message, false);
    throw e;
  }
}

/**
 * Preprocess an <img> element to a Float32Array in CHW format [3, 224, 224].
 * Matches the Python pipeline:
 *   Resize(224,224) → Grayscale(3ch) → ToTensor() → Normalize(ImageNet)
 */
function preprocessImage(imgEl) {
  const canvas = document.createElement("canvas");
  canvas.width  = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224); // RGBA 0-255

  const arr = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    const r    = data[i * 4]     / 255.0;
    const g    = data[i * 4 + 1] / 255.0;
    const b    = data[i * 4 + 2] / 255.0;
    // ITU-R 601 luma (matches PIL.Image.convert("L"))
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    arr[0 * 224 * 224 + i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    arr[1 * 224 * 224 + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    arr[2 * 224 * 224 + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return arr;
}

async function runOnnxInference(imgEl) {
  await ensureModel();
  const inputData = preprocessImage(imgEl);
  const inputTensor = new ort.Tensor("float32", inputData, [1, 3, 224, 224]);
  const results = await ortSession.run({ input: inputTensor });
  // Model outputs sigmoid probability → P(PNEUMONIA)
  return results["output"].data[0];
}

function showDemoResult(prob) {
  const pneumoniaPct = (prob * 100).toFixed(1);
  const normalPct    = ((1 - prob) * 100).toFixed(1);
  const isPneumonia  = prob >= 0.5;

  const badge = document.getElementById("resultBadge");
  badge.textContent = isPneumonia ? "PNEUMONIA" : "NORMAL";
  badge.className   = "result-badge " + (isPneumonia ? "result-pneumonia" : "result-normal");

  document.getElementById("pneumoniaPercent").textContent = pneumoniaPct + "%";
  document.getElementById("normalPercent").textContent    = normalPct    + "%";

  const pBar = document.getElementById("pneumoniaBar");
  const nBar = document.getElementById("normalBar");
  pBar.style.width = pneumoniaPct + "%";
  pBar.setAttribute("aria-valuenow", pneumoniaPct);
  nBar.style.width = normalPct + "%";
  nBar.setAttribute("aria-valuenow", normalPct);

  document.getElementById("demoResult").style.display = "block";
}

function resetDemo() {
  document.getElementById("uploadZone").style.display  = "block";
  document.getElementById("demoPreview").style.display = "none";
  document.getElementById("demoResult").style.display  = "none";
  document.getElementById("xrayInput").value           = "";
  setModelStatus(
    ortSession
      ? "Model loaded. Ready to analyze."
      : "Model not yet loaded — will load automatically on first analysis.",
    false
  );
}

window.addEventListener("DOMContentLoaded", () => {
  const input      = document.getElementById("xrayInput");
  const uploadZone = document.getElementById("uploadZone");
  const analyzeBtn = document.getElementById("analyzeBtn");
  if (!input) return; // demo section not present

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url        = URL.createObjectURL(file);
    const previewImg = document.getElementById("previewImg");
    previewImg.onload = () => URL.revokeObjectURL(url);
    previewImg.src    = url;
    uploadZone.style.display  = "none";
    document.getElementById("demoResult").style.display = "none";
    document.getElementById("demoPreview").style.display = "block";
  }

  input.addEventListener("change", (e) => handleFile(e.target.files[0]));

  uploadZone.addEventListener("dragover",  (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
  uploadZone.addEventListener("dragleave", ()  => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    handleFile(e.dataTransfer.files[0]);
  });

  analyzeBtn.addEventListener("click", async () => {
    const previewImg = document.getElementById("previewImg");
    if (!previewImg.src || previewImg.src === window.location.href) return;

    analyzeBtn.disabled     = true;
    analyzeBtn.innerHTML    =
      '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Analyzing…';

    try {
      const prob = await runOnnxInference(previewImg);
      showDemoResult(prob);
    } catch (err) {
      setModelStatus("Error during inference: " + err.message, false);
    } finally {
      analyzeBtn.disabled  = false;
      analyzeBtn.innerHTML = '<i class="bi bi-cpu me-1"></i>Analyze X-ray';
    }
  });
});
