import tempfile
import streamlit as st
from PIL import Image

from src.inference import predict

st.set_page_config(
    page_title="screen-recapture-detector",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Spot the Fake Photo")
st.caption(
    "Detect whether an image is a **REAL photograph** or a **PHOTO OF A SCREEN**."
)

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("About")

    st.write("""
This demo uses an EfficientNet-B0 model trained to classify images into:

✅ Real Photo

🖥️ Photo of a Screen
""")

    st.divider()

    st.write("**Model** : EfficientNet-B0")
    st.write("**Framework** : PyTorch")
    st.write("**Inference** : On-device CPU")

# ---------------- Tabs ---------------- #

tab1, tab2 = st.tabs(
    [
        "📁 Upload Image",
        "📷 Camera"
    ]
)

uploaded_file = None
camera_file = None

with tab1:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

with tab2:

    camera_file = st.camera_input(
        "Capture an image"
    )

image_source = uploaded_file if uploaded_file else camera_file

# ---------------- Prediction ---------------- #

if image_source is not None:

    image = Image.open(image_source).convert("RGB")

    st.image(
    image,
    caption="Selected Image",
    width=350
    )

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name

    with st.spinner("Analyzing image..."):

        label, probability, inference = predict(temp_path)

    st.divider()

    score = probability

    if label.lower() == "screen":

        st.error("🖥️ **🖥️ Prediction: Screen Recapture**")

    else:

        st.success("📷 **Prediction : REAL PHOTO**")

    st.progress(score)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Screen Probability",
            f"{score:.4f}"
        )

    with col2:

        st.metric(
            "Latency",
            f"{inference:.2f} ms"
        )

    if score > 0.90:

        st.success("Very High Confidence")

    elif score > 0.70:

        st.info("High Confidence")

    elif score > 0.50:

        st.warning("Moderate Confidence")

    else:

        st.info("Low Screen Probability")

    with st.expander("Prediction Details"):

        st.write(f"**Predicted Class :** {label.upper()}")
        st.write(f"**Screen Probability :** {score:.4f}")
        st.write(f"**Inference Time :** {inference:.2f} ms")
        st.write("**Model :** EfficientNet-B0")
        st.write("**Device :** CPU")

    if st.button("🔄 Analyze Another Image"):

        st.rerun()

st.divider()

st.caption(
    "SalesCode AI Take-Home • EfficientNet-B0 • PyTorch"
)