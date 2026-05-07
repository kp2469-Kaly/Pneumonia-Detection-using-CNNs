const repoUrl = "https://github.com/kp2469-Kaly/Pneumonia-Detection-using-CNNs";

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

window.addEventListener("DOMContentLoaded", () => {
  setRepoLinks();
  renderTech();
  renderAucBars();
  enableReveal();
});
