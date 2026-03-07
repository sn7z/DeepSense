import streamlit as st
import uuid
import os
import time
from backend.llm_explainer import explain_with_llm
# -----------------------------
# PAGE CONFIG (modern look)
# -----------------------------
st.set_page_config(
    page_title="DeepFake-Sense",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# IMPORT BACKEND PREDICTORS
# -----------------------------
from backend.audio import predict_audio
from backend.image import predict_image
from backend.video import predict_video

# -----------------------------
# IMPORT EXPLAINABILITY
# -----------------------------
from explainability.audio_explainability.audio_xai import explain_audio
from explainability.image_explainability.image_xai import explain_image
from explainability.video_explainability.video_xai import explain_video


# -----------------------------
# UTILS
# -----------------------------
def save_uploaded_file(uploaded_file, suffix):
    temp_name = f"cache/temp_{uuid.uuid4().hex}.{suffix}"
    with open(temp_name, "wb") as f:
        f.write(uploaded_file.read())
    return temp_name


def animated_progress(prob):
    bar = st.progress(0)
    for i in range(int(prob * 100)):
        time.sleep(0.01)
        bar.progress(i + 1)


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🔍 DeepSense")
st.sidebar.caption("DeepFake Detection & Explainability")

tab_image, tab_audio, tab_video = st.tabs(
    ["🖼 Image Detection", "🎵 Audio Detection", "🎥 Video Detection"]
)


# ==========================================================
# 🖼 IMAGE DETECTION
# ==========================================================
with tab_image:

    st.title("🖼 Image Deepfake Detection")

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_image, caption="Uploaded Image", width=350)

        temp_path = save_uploaded_file(uploaded_image, "jpg")

        label, prob = predict_image(temp_path)

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### **{label}**")
            st.markdown(f"Fake Probability: **{prob*100:.2f}%**")
            animated_progress(prob)

        st.markdown("---")
        st.subheader("🔎 Explainability (Grad-CAM)")

        fig, cam_img = explain_image(
            img_path=temp_path,
            last_conv_layer_name="block14_sepconv2_act"
        )

        st.pyplot(fig)
        
        import tempfile
        import cv2

        # Save GradCAM temporarily
        temp_cam = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_cam.name, cam_img)

        # evidence = """
        # Grad-CAM highlights facial regions such as mouth and eyes.
        # Deepfake images often contain blending artifacts in these areas.
        # """

        llm_text = explain_with_llm(
            modality="image",
            prediction=label,
            confidence=prob,
            # evidence=evidence,
            image_paths=[temp_cam.name]
        )

        st.markdown("### 🤖 AI Explanation")
        st.write(llm_text)

        os.remove(temp_path)

# ==========================================================
# 🎵 AUDIO DETECTION
# ==========================================================
with tab_audio:

    st.title("🎵 Audio Deepfake Detection")

    uploaded_audio = st.file_uploader(
        "Upload an Audio File",
        type=["wav", "mp3"]
    )

    if uploaded_audio:
        temp_path = save_uploaded_file(uploaded_audio, "wav")

        label, probs = predict_audio(temp_path)

        st.subheader("Prediction")
        st.markdown(f"### **{label}**")
        st.markdown(f"Fake Probability: **{probs['fake']*100:.2f}%**")
        animated_progress(probs["fake"])

        st.markdown("---")
        st.subheader("🔎 Explainability (Audio XAI)")

        xai_outputs = explain_audio(temp_path)

        st.plotly_chart(xai_outputs["waveform_fig"], use_container_width=True)
        st.plotly_chart(xai_outputs["mfcc_fig"], use_container_width=True)
        st.plotly_chart(xai_outputs["spectral_centroid_fig"], use_container_width=True)
        st.plotly_chart(xai_outputs["zcr_fig"], use_container_width=True)

        llm_text = explain_with_llm(
        modality="audio",
        prediction=label,
        confidence=probs["fake"],
        # evidence="The following graphs were analyzed to determine if the audio is real or fake.",
        image_paths=[
            xai_outputs["mfcc_img"],
            xai_outputs["spectral_img"],
            xai_outputs["zcr_img"]
            ]
        )

        st.markdown("### 🤖 AI Explanation")
        st.write(llm_text)

        os.remove(temp_path)

# ==========================================================
# 🎥 VIDEO DETECTION
# ==========================================================
with tab_video:

    st.title("🎥 Video Deepfake Detection")

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        temp_path = save_uploaded_file(uploaded_video, "mp4")

        label, prob = predict_video(temp_path)

        st.subheader("Prediction")
        st.markdown(f"### **{label}**")
        st.markdown(f"Fake Probability: **{prob*100:.2f}%**")
        animated_progress(prob)

        st.markdown("---")
        st.subheader("🔎 Explainability (Key Frames)")

        cam_frames = explain_video(
            video_path=temp_path,
            last_conv_layer_name="block14_sepconv2_act",
            max_frames=6
        )

        cols = st.columns(3)
        for i, frame in enumerate(cam_frames):
            with cols[i % 3]:
                st.image(frame, caption=f"Frame {i+1}", width=260)

        import tempfile
        import cv2

        frame_paths = []

        for frame in cam_frames[:3]:   # only send first 3 frames
            temp_frame = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(temp_frame.name, frame)
            frame_paths.append(temp_frame.name)

        # evidence = """
        # Grad-CAM highlights facial regions across multiple frames.
        # Temporal inconsistencies detected between frames.
        # """

        llm_text = explain_with_llm(
            modality="video",
            prediction=label,
            confidence=prob,
            # evidence=evidence,
            image_paths=frame_paths
        )

        st.markdown("### 🤖 AI Explanation")
        st.write(llm_text)
        
        os.remove(temp_path)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("© DeepFake-Sense | Explainable AI for Multimedia Deepfake Detection")
