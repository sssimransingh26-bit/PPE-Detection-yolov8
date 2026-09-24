import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="PPE Detection",
    page_icon="🦺"
)

st.title("🦺 PPE Detection using YOLOv8")
st.write("Upload a construction image to detect Personal Protective Equipment.")

# Load trained model
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a construction worker image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Convert image to NumPy
    img_array = np.array(image)

    # Run detection
    results = model(img_array)

    # Draw bounding boxes
    result_image = results[0].plot()

    st.subheader("PPE Detection Result")
    st.image(result_image, use_container_width=True)

    # Display detections
    st.subheader("Detected Objects")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            st.write(
                f"**{class_name}** → "
                f"Confidence: {confidence:.2%}"
            )

    else:
        st.warning("No objects detected.")