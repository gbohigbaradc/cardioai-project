# ============================================================
# CardioAI — Script 07: CNN Medical Imaging Module
# ============================================================
# Uses TorchXRayVision (DenseNet-121 + PSPNet) pretrained on
# 100,000+ chest X-rays to classify 18 pathologies, perform
# anatomical segmentation, compute cardiothoracic ratio,
# and generate Grad-CAM heatmaps for explainability.
#
# HOW TO RUN:
#   python 07_cnn_imaging.py
#
# OUTPUT:
#   outputs/cnn_sample_output.png  — demo on sample image
#   models/cnn_classifier.pt       — model weights cached
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("CardioAI — CNN Medical Imaging Module")
print("=" * 50)

# ── IMPORTS ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchxrayvision as xrv
    import skimage
    import skimage.io
    import skimage.transform
    print("✓ PyTorch and TorchXRayVision loaded")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Run: pip install torchxrayvision scikit-image torch torchvision")
    sys.exit(1)

# ── PATHOLOGY THRESHOLDS ─────────────────────────────────────
# Clinical thresholds for flagging abnormalities
# Based on optimal operating points from literature
THRESHOLDS = {
    "Cardiomegaly":       0.35,  # lower threshold — cardiac rehab priority
    "Effusion":           0.40,
    "Pneumonia":          0.30,
    "Atelectasis":        0.40,
    "Consolidation":      0.40,
    "Pneumothorax":       0.30,  # urgent — lower threshold
    "Edema":              0.35,
    "Emphysema":          0.40,
    "Fibrosis":           0.40,
    "Nodule":             0.45,
    "Mass":               0.40,
    "Infiltration":       0.40,
    "Pleural_Thickening": 0.40,
    "Hernia":             0.30,
}

SEVERITY = {
    (0.30, 0.50): "Mild",
    (0.50, 0.70): "Moderate",
    (0.70, 1.00): "Significant",
}

URGENT = {"Pneumothorax", "Mass", "Edema", "Effusion"}

CARDIAC_RELEVANT = {"Cardiomegaly", "Effusion", "Edema",
                    "Consolidation", "Infiltration"}

def get_severity(score):
    for (lo, hi), label in SEVERITY.items():
        if lo <= score < hi:
            return label
    return "Significant"


# ══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_xray_image(image_input):
    """
    Load and preprocess X-ray from file path, numpy array, or PIL Image.
    Returns normalised tensor ready for TorchXRayVision models.
    """
    from PIL import Image as PILImage

    if isinstance(image_input, str):
        img = skimage.io.imread(image_input)
    elif isinstance(image_input, PILImage.Image):
        img = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("image_input must be file path, PIL Image, or numpy array")

    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Normalise to [-1024, 1024] range expected by TorchXRayVision
    img = xrv.datasets.normalize(img, img.max())

    # Add channel dimension
    img = img[None, ...]

    # Resize and centre crop to 224x224
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    img_tensor = torch.from_numpy(img).float()

    return img_tensor


def classify_pathologies(img_tensor):
    """
    Run DenseNet-121 CNN to score all 18 pathologies.
    Returns dict of {pathology: score} sorted by score descending.
    """
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor[None, ...])

    scores = dict(zip(model.pathologies, outputs[0].detach().numpy().tolist()))

    # Sort by score descending
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return scores, model


def segment_anatomy(img_tensor):
    """
    Run PSPNet anatomical segmentation.
    Returns pixel-level masks for 14 anatomical structures.
    Targets: Left/Right Lung, Heart, Aorta, Spine, Clavicles etc.
    """
    try:
        seg_model = xrv.baseline_models.chestx_det.PSPNet()
        seg_model.eval()

        with torch.no_grad():
            output = seg_model(img_tensor[None, ...])

        masks = output[0].detach().numpy()
        targets = seg_model.targets

        return masks, targets
    except Exception as e:
        print(f"  Segmentation unavailable: {e}")
        return None, None


def compute_cardiothoracic_ratio(masks, targets):
    """
    Compute cardiothoracic ratio (CTR) from segmentation masks.
    CTR = max heart width / max chest width
    Normal CTR < 0.5. CTR >= 0.5 indicates Cardiomegaly.
    """
    if masks is None or targets is None:
        return None, None

    try:
        heart_idx  = targets.index("Heart")
        l_lung_idx = targets.index("Left Lung")
        r_lung_idx = targets.index("Right Lung")

        heart_mask  = masks[heart_idx]  > 0.5
        l_lung_mask = masks[l_lung_idx] > 0.5
        r_lung_mask = masks[r_lung_idx] > 0.5

        # Heart width: max horizontal extent of heart mask
        heart_cols = np.where(heart_mask.any(axis=0))[0]
        if len(heart_cols) < 2:
            return None, None
        heart_width = heart_cols[-1] - heart_cols[0]

        # Chest width: max horizontal extent of both lungs combined
        chest_mask = l_lung_mask | r_lung_mask | heart_mask
        chest_cols = np.where(chest_mask.any(axis=0))[0]
        if len(chest_cols) < 2:
            return None, None
        chest_width = chest_cols[-1] - chest_cols[0]

        ctr = heart_width / chest_width if chest_width > 0 else None
        cardiomegaly = ctr >= 0.50 if ctr is not None else None

        return ctr, cardiomegaly

    except (ValueError, IndexError) as e:
        return None, None


def generate_gradcam(img_tensor, model, target_pathology):
    """
    Generate Grad-CAM heatmap showing which image regions
    activated the CNN most for a given pathology.
    Returns heatmap as 2D numpy array (0-1 normalised).
    """
    model.eval()

    # Get pathology index
    pathologies = list(model.pathologies)
    if target_pathology not in pathologies:
        return None
    target_idx = pathologies.index(target_pathology)

    # Hook to capture gradients and activations from final conv layer
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the last dense block
    target_layer = model.model.features.denseblock4
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    img_input = img_tensor[None, ...].requires_grad_(True)
    outputs = model(img_input)

    # Backward pass for target pathology
    model.zero_grad()
    outputs[0, target_idx].backward()

    # Compute Grad-CAM
    grads   = gradients[0][0]       # [C, H, W]
    acts    = activations[0][0]     # [C, H, W]

    weights = grads.mean(dim=[1, 2])          # [C]
    cam     = (weights[:, None, None] * acts).sum(0)  # [H, W]
    cam     = F.relu(cam)
    cam     = cam.detach().numpy()

    # Normalise
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize to 224x224
    cam_resized = skimage.transform.resize(cam, (224, 224))

    fwd_handle.remove()
    bwd_handle.remove()

    return cam_resized


def analyse_flags(scores):
    """
    Identify flagged abnormalities with severity and urgency.
    Returns list of dicts sorted by urgency then score.
    """
    flags = []
    for pathology, score in scores.items():
        threshold = THRESHOLDS.get(pathology, 0.45)
        if score >= threshold:
            flags.append({
                "pathology": pathology,
                "score":     round(score, 3),
                "severity":  get_severity(score),
                "urgent":    pathology in URGENT,
                "cardiac":   pathology in CARDIAC_RELEVANT,
                "threshold": threshold,
            })

    # Sort: urgent first, then by score
    flags.sort(key=lambda x: (not x["urgent"], -x["score"]))
    return flags


def generate_report(scores, flags, ctr, cardiomegaly):
    """
    Generate structured clinical report text.
    """
    lines = []
    lines.append("=" * 56)
    lines.append("  CARDIOAI — CHEST X-RAY ANALYSIS REPORT")
    lines.append("  AI-Assisted Screening — Not a diagnostic replacement")
    lines.append("=" * 56)

    # Overall impression
    if not flags:
        lines.append("\nIMPRESSION: No significant findings detected.")
        lines.append("All pathology scores below clinical threshold.")
    else:
        urgent_flags = [f for f in flags if f["urgent"]]
        if urgent_flags:
            lines.append(f"\nIMPRESSION: URGENT — {len(urgent_flags)} urgent finding(s) detected.")
        else:
            lines.append(f"\nIMPRESSION: {len(flags)} finding(s) detected. Clinical review advised.")

    # Cardiothoracic ratio
    if ctr is not None:
        lines.append(f"\nCARDIOTHORACIC RATIO (CTR): {ctr:.3f}")
        if cardiomegaly:
            lines.append("  *** CARDIOMEGALY DETECTED — CTR >= 0.50 ***")
            lines.append("  Recommend: Echocardiogram, cardiology referral")
        else:
            lines.append("  Heart size within normal limits (CTR < 0.50)")

    # Flagged findings
    if flags:
        lines.append(f"\nFINDINGS ({len(flags)} detected):")
        for f in flags:
            urgency = " [URGENT]" if f["urgent"] else ""
            cardiac = " [CARDIAC]" if f["cardiac"] else ""
            lines.append(
                f"  {'!' if f['urgent'] else '-'} {f['pathology']:<25} "
                f"Score: {f['score']:.3f}  {f['severity']}{urgency}{cardiac}"
            )

    # All scores
    lines.append("\nFULL PATHOLOGY SCORES:")
    for pathology, score in scores.items():
        flagged = "**" if any(f["pathology"] == pathology for f in flags) else "  "
        lines.append(f"  {flagged} {pathology:<25} {score:.4f}")

    lines.append("\n" + "=" * 56)
    lines.append("  IMPORTANT: This is an AI screening tool.")
    lines.append("  All findings must be confirmed by a radiologist.")
    lines.append("  Do not act on this report without clinical review.")
    lines.append("=" * 56)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# DEMO RUN — generates sample output if no X-ray provided
# ══════════════════════════════════════════════════════════════

def create_demo_xray():
    """Create a synthetic chest X-ray-like image for testing."""
    img = np.zeros((512, 512), dtype=np.float32)

    # Background: dark (lung fields)
    img[:, :] = 20

    # Chest wall (bright outer edges)
    img[:, :40]  = 180
    img[:, 472:] = 180
    img[:30, :]  = 160
    img[480:, :] = 200

    # Right lung field (brighter = slightly opaque)
    from skimage.draw import ellipse
    rr, cc = ellipse(250, 150, 180, 110)
    img[rr, cc] = 60

    # Left lung field
    rr, cc = ellipse(250, 370, 180, 110)
    img[rr, cc] = 55

    # Heart shadow (bright — denser)
    rr, cc = ellipse(280, 260, 110, 90)
    img[rr, cc] = 140

    # Spine
    img[80:460, 240:272] = 200

    # Diaphragm
    img[390:410, 50:470] = 190

    # Add noise
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255)

    return img.astype(np.uint8)


def run_demo():
    print("\nRunning demo analysis on synthetic chest X-ray...")
    print("-" * 50)

    # Create demo image
    demo_img = create_demo_xray()
    skimage.io.imsave("outputs/demo_xray.png", demo_img)

    # Process
    print("Step 1: Loading and preprocessing image...")
    img_tensor = load_xray_image(demo_img)
    print(f"  Image tensor shape: {img_tensor.shape}")

    print("Step 2: Running DenseNet-121 pathology classification...")
    scores, model = classify_pathologies(img_tensor)
    print(f"  Classified {len(scores)} pathologies")

    print("Step 3: Running anatomical segmentation...")
    masks, targets = segment_anatomy(img_tensor)
    if masks is not None:
        print(f"  Segmented {len(targets)} anatomical structures")
        print(f"  Structures: {', '.join(targets[:6])}...")
    else:
        print("  Segmentation not available in this environment")

    print("Step 4: Computing cardiothoracic ratio...")
    ctr, cardiomegaly = compute_cardiothoracic_ratio(masks, targets)
    if ctr:
        print(f"  CTR = {ctr:.3f} ({'CARDIOMEGALY' if cardiomegaly else 'Normal'})")
    else:
        print("  CTR not computable (segmentation unavailable)")

    print("Step 5: Identifying flagged abnormalities...")
    flags = analyse_flags(scores)
    print(f"  {len(flags)} findings flagged")

    print("Step 6: Generating Grad-CAM for top finding...")
    top_pathology = flags[0]["pathology"] if flags else list(scores.keys())[0]
    gradcam = generate_gradcam(img_tensor, model, top_pathology)
    if gradcam is not None:
        print(f"  Grad-CAM generated for: {top_pathology}")

    # ── Generate visualisation ────────────────────────────────
    print("\nGenerating output visualisation...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0D1B2A")
    for ax in axes.flat:
        ax.set_facecolor("#0D1B2A")

    img_display = img_tensor[0].numpy()

    # 1. Original X-ray
    axes[0, 0].imshow(img_display, cmap="gray")
    axes[0, 0].set_title("Chest X-Ray Input", color="white", fontsize=12, pad=8)
    axes[0, 0].axis("off")

    # 2. Segmentation overlay
    if masks is not None:
        overlay = np.stack([img_display] * 3, axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        colors_map = {
            "Heart":      [1.0, 0.2, 0.2],
            "Left Lung":  [0.2, 0.6, 1.0],
            "Right Lung": [0.2, 0.8, 0.4],
            "Aorta":      [1.0, 0.8, 0.0],
        }
        for struct, color in colors_map.items():
            if struct in targets:
                idx  = targets.index(struct)
                mask = (skimage.transform.resize(masks[idx], (224, 224)) > 0.5)
                for c in range(3):
                    overlay[:, :, c] = np.where(mask, overlay[:, :, c] * 0.4 + color[c] * 0.6, overlay[:, :, c])
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title("Anatomical Segmentation", color="white", fontsize=12, pad=8)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=s) for s, c in colors_map.items()]
        axes[0, 1].legend(handles=legend_elements, loc="lower right",
                          fontsize=7, framealpha=0.7, facecolor="#1a1a2e", labelcolor="white")
    else:
        axes[0, 1].imshow(img_display, cmap="gray")
        axes[0, 1].set_title("Segmentation (unavailable)", color="gray", fontsize=12, pad=8)
    axes[0, 1].axis("off")

    # 3. Grad-CAM overlay
    if gradcam is not None:
        heatmap = cm.jet(gradcam)[:, :, :3]
        img_norm = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        img_rgb  = np.stack([img_norm] * 3, axis=-1)
        blended  = img_rgb * 0.5 + heatmap * 0.5
        axes[0, 2].imshow(blended)
        axes[0, 2].set_title(f"Grad-CAM: {top_pathology}", color="white", fontsize=12, pad=8)
    else:
        axes[0, 2].imshow(img_display, cmap="gray")
        axes[0, 2].set_title("Grad-CAM (unavailable)", color="gray", fontsize=12, pad=8)
    axes[0, 2].axis("off")

    # 4. Top pathology scores bar chart
    top_n = 10
    top_scores = list(scores.items())[:top_n]
    pathologies = [p for p, _ in top_scores]
    values      = [s for _, s in top_scores]
    flagged     = [any(f["pathology"] == p for f in flags) for p in pathologies]
    bar_colors  = ["#E63946" if f else "#00B4D8" for f in flagged]

    bars = axes[1, 0].barh(range(len(pathologies)), values, color=bar_colors, height=0.6)
    axes[1, 0].set_yticks(range(len(pathologies)))
    axes[1, 0].set_yticklabels(pathologies, color="white", fontsize=9)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].axvline(x=0.4, color="yellow", linestyle="--", alpha=0.5, linewidth=1, label="Threshold")
    axes[1, 0].set_title("Pathology Scores (red = flagged)", color="white", fontsize=12, pad=8)
    axes[1, 0].tick_params(colors="white")
    axes[1, 0].set_facecolor("#0D1B2A")
    for spine in axes[1, 0].spines.values():
        spine.set_color("#2C3E50")
    axes[1, 0].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    # 5. Flagged findings summary
    axes[1, 1].axis("off")
    if flags:
        text = "FLAGGED FINDINGS\n" + "─" * 30 + "\n\n"
        for f in flags[:6]:
            urgency = " ⚠ URGENT" if f["urgent"] else ""
            cardiac = " ♥ CARDIAC" if f["cardiac"] else ""
            text += f"{'!' if f['urgent'] else '•'} {f['pathology']}\n"
            text += f"  Score: {f['score']:.3f}  [{f['severity']}]{urgency}{cardiac}\n\n"
        if ctr:
            text += f"\nCTR: {ctr:.3f}"
            text += " — CARDIOMEGALY" if cardiomegaly else " — Normal"
    else:
        text = "NO FINDINGS\n\nAll pathology scores\nbelow threshold.\n\nScan appears clear."

    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes,
                    color="white", fontsize=9, va="top", ha="left",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#1A2A3A", alpha=0.8))
    axes[1, 1].set_title("Clinical Findings", color="white", fontsize=12, pad=8)

    # 6. CTR visualisation
    axes[1, 2].axis("off")
    if ctr:
        # Draw CTR gauge
        ax = axes[1, 2]
        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "white", lw=2, alpha=0.3)

        ctr_angle = np.pi * (1 - min(ctr / 0.8, 1.0))
        needle_x = [0, 0.8 * np.cos(ctr_angle)]
        needle_y = [0, 0.8 * np.sin(ctr_angle)]
        color = "#E63946" if cardiomegaly else "#06D6A0"
        ax.plot(needle_x, needle_y, color=color, lw=3)
        ax.add_patch(plt.Circle((0, 0), 0.05, color=color))

        ax.text(0, -0.25, f"CTR = {ctr:.3f}", ha="center", color="white", fontsize=14, fontweight="bold")
        ax.text(0, -0.45, "CARDIOMEGALY" if cardiomegaly else "Normal", ha="center",
                color=color, fontsize=11, fontweight="bold")
        ax.text(-0.95, -0.1, "0.0", color="white", fontsize=8)
        ax.text(0.80, -0.1, "0.8+", color="white", fontsize=8)
        ax.text(-0.1, 0.85, "0.4", color="white", fontsize=8)
        ax.axvline(x=np.cos(np.pi * 0.375), ymin=0, ymax=0.5, color="yellow", alpha=0.4, lw=1.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.6, 1.1)
        ax.set_title("Cardiothoracic Ratio (CTR)", color="white", fontsize=12, pad=8)
    else:
        axes[1, 2].text(0.5, 0.5, "CTR Not\nComputable\n\n(Segmentation\nunavailable)",
                        ha="center", va="center", color="gray", fontsize=12,
                        transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("Cardiothoracic Ratio", color="gray", fontsize=12, pad=8)

    plt.suptitle("CardioAI — CNN Medical Imaging Analysis", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/cnn_sample_output.png", dpi=150, bbox_inches="tight",
                facecolor="#0D1B2A", edgecolor="none")
    plt.close()
    print("  Visualisation saved: outputs/cnn_sample_output.png")

    # Print report
    report = generate_report(scores, flags, ctr, cardiomegaly)
    print("\n" + report)

    # Save report
    with open("outputs/cnn_sample_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved: outputs/cnn_sample_report.txt")
    print("\n✓ CNN Imaging Module demo complete")


if __name__ == "__main__":
    run_demo()
