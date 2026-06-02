const repoUrl = "https://github.com/kp2469-Kaly/Pneumonia-Detection-using-CNNs";
const ONNX_MODEL_URL = "assets/densenet121_best.onnx";
const IMAGE_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

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

let uploadedImageDataUrl = "";
let onnxSessionPromise = null;

function setRepoLinks() {
  const githubButton = document.getElementById("githubLink");
  const footerGithubLink = document.getElementById("footerGithubLink");

  if (repoUrl && repoUrl !== "#") {
    if (githubButton) githubButton.href = repoUrl;
    if (footerGithubLink) footerGithubLink.href = repoUrl;
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

function setStatus(message, isError = false) {
  const statusNode = document.getElementById("modelStatus");
  if (!statusNode) return;
  statusNode.textContent = `Model status: ${message}`;
  statusNode.classList.toggle("status-error", isError);
}

function setResultVisibility(visible) {
  const card = document.getElementById("predictionResult");
  if (!card) return;
  card.classList.toggle("d-none", !visible);
}

function resetPredictionUi() {
  const preview = document.getElementById("previewImage");
  const placeholder = document.getElementById("previewPlaceholder");
  const uploader = document.getElementById("uploadXray");
  const predictBtn = document.getElementById("predictBtn");

  if (preview) {
    preview.src = "";
    preview.classList.add("d-none");
  }
  if (placeholder) placeholder.classList.remove("d-none");
  if (uploader) uploader.value = "";
  if (predictBtn) predictBtn.disabled = true;

  uploadedImageDataUrl = "";
  setResultVisibility(false);
  setStatus("waiting for image.");
}

async function loadOnnxSession() {
  if (onnxSessionPromise) return onnxSessionPromise;

  if (typeof ort === "undefined") {
    throw new Error("ONNX Runtime failed to load. Check your network or script include.");
  }

  onnxSessionPromise = ort.InferenceSession.create(ONNX_MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
  return onnxSessionPromise;
}

function dataUrlToImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read uploaded image."));
    image.src = dataUrl;
  });
}

function preprocessToTensor(image) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;

  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const pixels = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const hw = IMAGE_SIZE * IMAGE_SIZE;
  const tensorData = new Float32Array(3 * hw);

  for (let i = 0; i < hw; i += 1) {
    const base = i * 4;
    const r = pixels[base] / 255;
    const g = pixels[base + 1] / 255;
    const b = pixels[base + 2] / 255;

    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    tensorData[i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    tensorData[hw + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    tensorData[(2 * hw) + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return new ort.Tensor("float32", tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function extractProbability(output) {
  if (!output || !output.data || !output.data.length) {
    throw new Error("Model output is empty.");
  }
  const probability = Number(output.data[0]);
  return Math.max(0, Math.min(1, probability));
}

function renderPrediction(probability) {
  const predictedClass = probability >= 0.5 ? "PNEUMONIA" : "NORMAL";
  const pPneumonia = document.getElementById("pPneumonia");
  const pNormal = document.getElementById("pNormal");
  const label = document.getElementById("predictedClass");
  const fill = document.getElementById("confidenceFill");

  if (label) label.textContent = predictedClass;
  if (pPneumonia) pPneumonia.textContent = probability.toFixed(4);
  if (pNormal) pNormal.textContent = (1 - probability).toFixed(4);
  if (fill) fill.style.width = `${(Math.max(probability, 1 - probability) * 100).toFixed(1)}%`;

  setResultVisibility(true);
}

function setupInference() {
  const uploader = document.getElementById("uploadXray");
  const preview = document.getElementById("previewImage");
  const placeholder = document.getElementById("previewPlaceholder");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");

  if (!uploader || !preview || !placeholder || !predictBtn || !clearBtn) {
    return;
  }

  uploader.addEventListener("change", () => {
    const [file] = uploader.files || [];
    if (!file) {
      resetPredictionUi();
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    uploadedImageDataUrl = objectUrl;
    preview.src = objectUrl;
    preview.classList.remove("d-none");
    placeholder.classList.add("d-none");
    predictBtn.disabled = false;
    setResultVisibility(false);
    setStatus("image loaded. Ready to predict.");
  });

  clearBtn.addEventListener("click", () => {
    resetPredictionUi();
  });

  predictBtn.addEventListener("click", async () => {
    if (!uploadedImageDataUrl) {
      setStatus("please upload an image first.", true);
      return;
    }

    try {
      predictBtn.disabled = true;
      setStatus("loading model...");
      const session = await loadOnnxSession();

      setStatus("preprocessing image...");
      const image = await dataUrlToImage(uploadedImageDataUrl);
      const inputTensor = preprocessToTensor(image);

      setStatus("running inference...");
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      const outputs = await session.run({ [inputName]: inputTensor });
      const probability = extractProbability(outputs[outputName]);

      renderPrediction(probability);
      setStatus("prediction complete.");
    } catch (error) {
      console.error(error);
      setStatus(
        "inference failed. Ensure assets/densenet121_best.onnx exists and was exported from your checkpoint.",
        true
      );
    } finally {
      predictBtn.disabled = false;
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setRepoLinks();
  renderTech();
  renderAucBars();
  enableReveal();
  setupInference();
});
