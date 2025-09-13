import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="🌿 Crop Disease Detector",
    page_icon="🌿",
    layout="centered",
)


class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def _normalize_cam(self, cam_batch):
        cams = []
        for i in range(cam_batch.size(0)):
            c = cam_batch[i, 0]
            c = c - c.min()
            denom = (c.max() + 1e-8)
            c = c / denom
            cams.append(c.cpu().numpy())
        return np.stack(cams, axis=0)

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        self.model.zero_grad(set_to_none=True)
        loss = logits[torch.arange(logits.size(0)), class_idx].sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return self._normalize_cam(cam)

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def tensor_to_rgb01(img_tensor):
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.detach().cpu() * inv_std + inv_mean
    img = img.clamp(0,1).permute(1,2,0).numpy().astype(np.float32)
    return img

def overlay_cam_on_image(rgb01, cam01, alpha=0.35):
    cmap = plt.cm.jet(cam01)[..., :3]
    overlay = (1 - alpha) * rgb01 + alpha * cmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

def get_target_layer_for_gradcam(model, arch: str):
    if arch == "mobilenetv2":
        return model.features[-1]
    elif arch == "resnet50":
        return model.layer4[-1]
    else:
        raise ValueError("Unsupported arch for Grad-CAM")


@st.cache_resource
def load_model_and_labels(weights_path: str, device: str = "cpu"):
    device = torch.device(device)
    ckpt = torch.load(weights_path, map_location=device)

    arch = ckpt.get("arch", "mobilenetv2")
    classes = ckpt["classes"]

    if arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, len(classes))
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    idx_to_class = {i: c for i, c in enumerate(classes)}
    return model, idx_to_class, device, arch

def get_preprocess(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def predict_topk(model, device, img_pil: Image.Image, idx_to_class, k: int = 3):
    preprocess = get_preprocess(224)
    x = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=k)
    results = []
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        results.append((idx_to_class[idx], float(p)))
    return results

def run_gradcam(model, device, arch: str, img_pil: Image.Image):
    rgb_img = ImageOps.exif_transpose(img_pil).convert("RGB").resize((224,224))
    preprocess = get_preprocess(224)
    input_tensor = preprocess(rgb_img).unsqueeze(0).to(device)

    target_layer = get_target_layer_for_gradcam(model, arch)
    cam_engine = SimpleGradCAM(model, target_layer)
    cam_map = cam_engine(input_tensor)[0]
    rgb01 = tensor_to_rgb01(input_tensor[0])
    overlay = overlay_cam_on_image(rgb01, cam_map)
    cam_engine.close()
    return Image.fromarray(overlay)


st.sidebar.title("⚙️ Configuration")
st.sidebar.caption("Point to your trained weights. The app will auto-load labels.")

weights_default = str("/content/models/mobilenetv2_finetuned_best.pt")
weights_path = st.sidebar.text_input("Weights path (.pt)", value=weights_default)
device_choice = "cuda" if torch.cuda.is_available() else "cpu"
device = st.sidebar.selectbox("Device", [device_choice, "cpu"], index=0)

show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=False)


st.title("🌿 Crop Disease Detector")
st.write(
    "Upload a leaf photo and the model will predict the most likely disease (or healthy). "
    "This is an educational demo trained on PlantVillage; real field conditions may vary."
)

try:
    model, idx_to_class, device, arch = load_model_and_labels(weights_path, device=device)
    num_classes = len(idx_to_class)
except Exception as e:
    st.error(f"Could not load model/weights.\n\n{e}")
    st.stop()

uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            preds = predict_topk(model, device, img, idx_to_class, k=3)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    st.subheader("Top-3 predictions")
    for label, p in preds:
        st.write(f"- **{label}** — {p*100:.1f}%")

    if show_gradcam:
        with st.spinner("Generating Grad-CAM..."):
            try:
                overlay = run_gradcam(model, device, arch, img)
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                bio = io.BytesIO()
                overlay.save(bio, format="PNG")
                st.download_button("Download overlay", data=bio.getvalue(),
                                   file_name="gradcam_overlay.png", mime="image/png")
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")

st.caption("⚠️ Not a diagnostic tool. For field use, further validation on real-world images is needed.")