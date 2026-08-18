import json
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path("fault_classifier.pt")
CLASS_CANDIDATES = [Path("classes_names.json"), Path("class_names.json")]
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explanation + fix-step templates per fault class
FAULT_INFO = {
    "missing_hole": {
        "explanation": "A required drill hole is missing from the board, which can prevent a component lead or via from being placed correctly.",
        "fix_steps": ["Re-check the PCB layout against the schematic.", "Re-drill or re-fabricate the board with the correct hole placement."],
    },
    "mouse_bite": {
        "explanation": "A small semicircular notch is cut into a copper pad or trace, which can thin the copper enough to cause an intermittent or open connection.",
        "fix_steps": ["Inspect the trace under magnification.", "Reflow solder or bridge the trace with a short jumper wire if the defect breaks continuity."],
    },
    "open_circuit": {
        "explanation": "A trace that should carry current is broken somewhere along its path, so the circuit is incomplete.",
        "fix_steps": ["Use a multimeter continuity test to locate the break.", "Bridge the gap with a solder jumper or repair wire."],
    },
    "short": {
        "explanation": "Two traces or pads that should be electrically separate are touching, which can cause excess current draw or damage components.",
        "fix_steps": ["Visually inspect for solder bridges.", "Use a multimeter to confirm the short, then carefully cut or desolder the bridging copper."],
    },
    "spur": {
        "explanation": "An unwanted stray piece of copper extends off a trace, which risks shorting to a nearby trace or pad.",
        "fix_steps": ["Inspect closely for the stray copper.", "Trim it away carefully with a hobby knife or re-etch the board."],
    },
    "spurious_copper": {
        "explanation": "Extra copper exists where it shouldn't, which can create unintended connections.",
        "fix_steps": ["Identify which nets the extra copper touches.", "Remove the excess copper or re-fabricate the board."],
    },
}

st.set_page_config(page_title="AI Engineering Troubleshooting Assistant")


@st.cache_resource
def load_model():
    classes_path = next((p for p in CLASS_CANDIDATES if p.exists()), None)
    if not MODEL_PATH.exists() or classes_path is None:
        return None, None

    class_names = json.loads(classes_path.read_text())

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def classify_fault(model, class_names, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_idx = int(probs.argmax())
    return class_names[top_idx], float(probs[top_idx])


def translate_to_nepali(text: str) -> str:
    return GoogleTranslator(source="en", target="ne").translate(text)


def text_to_speech_nepali(text: str) -> bytes:
    tts = gTTS(text=text, lang="ne")
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


#UIs
st.title("🔧 AI Engineering Troubleshooting Assistant")
st.caption("Fine-tuned locally via transfer learning (ResNet18) on PCB fault images.")

model, class_names = load_model()

if model is None:
    st.error(
        "No trained model found. Run `python dataset_prepare.py` then "
        "`python train.py` first to produce fault_classifier.pt."
    )
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or capture from camera")
image_file = uploaded or camera_file

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded setup", use_container_width=True)

    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Classifying..."):
            pred_class, confidence = classify_fault(model, class_names, image)
            st.session_state["pred_class"] = pred_class
            st.session_state["confidence"] = confidence

if "pred_class" in st.session_state:
    pred_class = st.session_state["pred_class"]
    confidence = st.session_state["confidence"]
    info = FAULT_INFO.get(pred_class, {"explanation": "No details available for this class.", "fix_steps": []})

    st.subheader("⚠️ Detected Fault")
    st.write(f"**{pred_class.replace('_', ' ').title()}** (confidence: {confidence:.1%})")

    st.subheader("🧠 Explanation")
    st.write(info["explanation"])

    st.subheader("🛠️ Suggested Fix Steps")
    for i, step in enumerate(info["fix_steps"], 1):
        st.write(f"{i}. {step}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🇳🇵 Translate to Nepali"):
            with st.spinner("Translating..."):
                st.session_state["nepali_text"] = translate_to_nepali(info["explanation"])

    if "nepali_text" in st.session_state:
        st.write(st.session_state["nepali_text"])
        with col2:
            if st.button("🔊 Speak in Nepali"):
                with st.spinner("Generating audio..."):
                    audio_bytes = text_to_speech_nepali(st.session_state["nepali_text"])
                    st.audio(audio_bytes, format="audio/mp3")