const clips = [
  {
    id: "cartwheel-18",
    category: "tracking",
    title: "Cartwheel tracking, motion 18",
    kicker: "Tracking",
    video: "./public/media/cartwheel_18.mp4",
    badges: ["FB", "cartwheel", "0.95 / 0.05 mix"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN 95% + cartwheel 5%",
      Checkpoint: "lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10",
      Motion: "soma_cartwheel_near10s.pkl / id 18",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.tracking_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10 \\
  --data-path /data/xue/bfmzero/data/soma_cartwheel_near10s.pkl \\
  --device cuda:0 \\
  --motion-list 18 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "cartwheel-22",
    category: "tracking",
    title: "Cartwheel tracking, motion 22",
    kicker: "Tracking",
    video: "./public/media/cartwheel_22.mp4",
    badges: ["FB", "cartwheel", "tracking"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN 95% + cartwheel 5%",
      Checkpoint: "lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10",
      Motion: "soma_cartwheel_near10s.pkl / id 22",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.tracking_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10 \\
  --data-path /data/xue/bfmzero/data/soma_cartwheel_near10s.pkl \\
  --device cuda:0 \\
  --motion-list 22 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "cartwheel-29",
    category: "tracking",
    title: "Cartwheel tracking, motion 29",
    kicker: "Tracking",
    video: "./public/media/cartwheel_29.mp4",
    badges: ["FB", "cartwheel", "tracking"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN 95% + cartwheel 5%",
      Checkpoint: "lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10",
      Motion: "soma_cartwheel_near10s.pkl / id 29",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.tracking_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10 \\
  --data-path /data/xue/bfmzero/data/soma_cartwheel_near10s.pkl \\
  --device cuda:0 \\
  --motion-list 29 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "lafan-6",
    category: "tracking",
    title: "LAFAN tracking, motion 6",
    kicker: "Tracking",
    video: "./public/media/lafan_6.mp4",
    badges: ["FB", "LAFAN", "tracking"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN",
      Checkpoint: "formal_8gpu_mimiclite_dc_wandb",
      Motion: "lafan_29dof_10s-clipped.pkl / id 6",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.tracking_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb \\
  --data-path /data/xue/bfmzero/data/lafan_29dof_10s-clipped.pkl \\
  --device cuda:0 \\
  --motion-list 6 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "sit-ground",
    category: "prompts",
    title: "Reward prompt: sit to ground",
    kicker: "Prompted Behavior",
    video: "./public/media/sitonground.mp4",
    badges: ["FB", "reward", "sit"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN",
      Checkpoint: "formal_8gpu_mimiclite_dc_wandb",
      Prompt: "sitonground",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.reward_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb \\
  --device cuda:0 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "raise-arms",
    category: "prompts",
    title: "Reward prompt: raise arms",
    kicker: "Prompted Behavior",
    video: "./public/media/raisearms_l_l.mp4",
    badges: ["FB", "reward", "arms"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN",
      Checkpoint: "formal_8gpu_mimiclite_dc_wandb",
      Prompt: "raisearms-l-l",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.reward_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb \\
  --device cuda:0 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "crouch",
    category: "prompts",
    title: "Reward prompt: crouch",
    kicker: "Prompted Behavior",
    video: "./public/media/crouch_025.mp4",
    badges: ["FB", "reward", "crouch"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN",
      Checkpoint: "formal_8gpu_mimiclite_dc_wandb",
      Prompt: "crouch-0.25",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.reward_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb \\
  --device cuda:0 \\
  --save-mp4 \\
  --headless`,
  },
  {
    id: "goal",
    category: "goals",
    title: "Goal-conditioned locomotion",
    kicker: "Goal Control",
    video: "./public/media/goal_mjlab.mp4",
    badges: ["FB", "goal", "navigation"],
    metrics: {
      Policy: "FB checkpoint",
      Dataset: "LAFAN",
      Checkpoint: "formal_8gpu_mimiclite_dc_wandb",
      Rollout: "goal_inference",
      Robot: "Unitree G1, 29 DoF",
    },
    command:
`uv run python -m humanoidverse.goal_inference \\
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb \\
  --device cuda:0 \\
  --save-mp4 \\
  --headless`,
  },
];

const tabs = Array.from(document.querySelectorAll(".tab"));
const clipGrid = document.getElementById("clip-grid");
const clipTitle = document.getElementById("clip-title");
const clipKicker = document.getElementById("clip-kicker");
const clipBadges = document.getElementById("clip-badges");
const clipCount = document.getElementById("clip-count");
const galleryTitle = document.getElementById("gallery-title");
const metricGrid = document.getElementById("metric-grid");
const mainVideo = document.getElementById("main-video");
const missing = document.getElementById("media-missing");
const commandText = document.getElementById("command-text");
const copyCommand = document.getElementById("copy-command");

let activeCategory = "tracking";
let activeClip = clips.find((clip) => clip.category === activeCategory);

function categoryTitle(category) {
  if (category === "tracking") return "Tracking Clips";
  if (category === "prompts") return "Reward Prompt Clips";
  return "Goal Clips";
}

function renderBadges(clip) {
  clipBadges.innerHTML = "";
  clip.badges.forEach((badge) => {
    const node = document.createElement("span");
    node.className = "badge";
    node.textContent = badge;
    clipBadges.appendChild(node);
  });
}

function renderMetrics(clip) {
  metricGrid.innerHTML = "";
  Object.entries(clip.metrics).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "metric";
    row.innerHTML = `<span>${label}</span><span>${value}</span>`;
    metricGrid.appendChild(row);
  });
}

function setMainVideo(clip) {
  missing.classList.remove("is-visible");
  mainVideo.src = clip.video;
  mainVideo.load();
  mainVideo.play().catch(() => {});
}

function selectClip(clip) {
  activeClip = clip;
  clipTitle.textContent = clip.title;
  clipKicker.textContent = clip.kicker;
  commandText.textContent = clip.command;
  renderBadges(clip);
  renderMetrics(clip);
  setMainVideo(clip);
  document.querySelectorAll(".clip-card").forEach((card) => {
    card.classList.toggle("is-active", card.dataset.clipId === clip.id);
  });
}

function renderGrid() {
  const filtered = clips.filter((clip) => clip.category === activeCategory);
  clipGrid.innerHTML = "";
  clipCount.textContent = `${filtered.length} clips`;
  galleryTitle.textContent = categoryTitle(activeCategory);
  filtered.forEach((clip) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "clip-card";
    button.dataset.clipId = clip.id;
    button.innerHTML = `
      <div class="clip-thumb">
        <video src="${clip.video}" muted loop playsinline preload="metadata"></video>
      </div>
      <div class="clip-copy">
        <strong>${clip.title}</strong>
        <span>${clip.badges.join(" / ")}</span>
      </div>
    `;
    button.addEventListener("mouseenter", () => {
      const video = button.querySelector("video");
      video.play().catch(() => {});
    });
    button.addEventListener("mouseleave", () => {
      const video = button.querySelector("video");
      video.pause();
    });
    button.addEventListener("click", () => selectClip(clip));
    clipGrid.appendChild(button);
  });
  selectClip(filtered[0]);
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    activeCategory = tab.dataset.category;
    tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    renderGrid();
  });
});

mainVideo.addEventListener("error", () => {
  missing.classList.add("is-visible");
});

copyCommand.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(activeClip.command);
    copyCommand.textContent = "Copied";
    window.setTimeout(() => {
      copyCommand.textContent = "Copy";
    }, 1000);
  } catch {
    copyCommand.textContent = "Select";
    window.setTimeout(() => {
      copyCommand.textContent = "Copy";
    }, 1000);
  }
});

renderGrid();
