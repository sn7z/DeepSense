from google import genai
import os
import time
import streamlit as st
import threading

# -----------------------------
# RATE LIMIT SETTINGS
# -----------------------------
RATE_LIMIT_SECONDS = 5   # per-user (~12 RPM)
GLOBAL_RATE_LIMIT = 4    # ~15 RPM safe globally

# Global tracker (shared across users)
LAST_GLOBAL_CALL = 0
lock = threading.Lock()


def explain_with_llm(modality, prediction, confidence, image_paths=None):

    global LAST_GLOBAL_CALL

    # -----------------------------
    # PER-USER RATE LIMIT
    # -----------------------------
    if "last_called" not in st.session_state:
        st.session_state.last_called = 0

    current_time = time.time()

    if current_time - st.session_state.last_called < RATE_LIMIT_SECONDS:
        return "⚠️ Too many requests. Please wait a few seconds."

    # -----------------------------
    # GLOBAL RATE LIMIT (MULTI-USER SAFE)
    # -----------------------------
    with lock:
        current_time = time.time()

        if current_time - LAST_GLOBAL_CALL < GLOBAL_RATE_LIMIT:
            return "⚠️ Server busy (rate limit reached). Try again shortly."

        LAST_GLOBAL_CALL = current_time

    # update user timestamp AFTER passing global check
    st.session_state.last_called = time.time()

    # -----------------------------
    # API KEY CHECK
    # -----------------------------
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ API key not set"

    client = genai.Client(api_key=api_key)

    # -----------------------------
    # INPUT HANDLING
    # -----------------------------
    if image_paths is None:
        image_paths = []

    # -----------------------------
    # PROMPT
    # -----------------------------
    prompt = f"""
You are an AI forensic analyst helping users understand deepfake detection results.

Media Type: {modality}
Prediction: {prediction}
Fake Prediction probability/Confidence: {confidence:.2f}

Explain clearly in simple language WHY the media might be {prediction}.
Refer to the visual evidence if available.
"""

    contents = []

    # -----------------------------
    # UPLOAD IMAGE FILES (paths → bytes)
    # -----------------------------
    for img_path in image_paths:
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            # Determine mime type from extension
            ext = os.path.splitext(img_path)[1].lower()
            mime_map = {
                ".png":  "image/png",
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/png")

            with open(img_path, "rb") as f:
                image_bytes = f.read()

            # Use the upload API correctly — config dict avoids kwarg issues
            # across different google-genai SDK minor versions
            uploaded_file = client.files.upload(
                path=img_path,
                config={"mime_type": mime_type},
            )
            contents.append(uploaded_file)
        except TypeError:
            # Fallback for older SDK versions that don't support config=
            import io
            from google.genai import types
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            contents.append(part)
        except Exception:
            # Skip unreadable files silently
            continue

    contents.append(prompt)

    # -----------------------------
    # GEMINI CALL
    # -----------------------------
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=contents
        )
        return response.text

    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"
