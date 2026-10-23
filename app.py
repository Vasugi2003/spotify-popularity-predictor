import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spotify Popularity Predictor", page_icon="🎵", layout="wide")
MODEL_PATH = "best_model.joblib"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def is_classifier(model):
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    return getattr(estimator, "_estimator_type", None) == "classifier"

st.markdown("""
<style>
.main-title {font-size:2.25rem;font-weight:800;margin-bottom:.2rem}
.subtitle {color:#9aa4b2;font-size:1.05rem;margin-bottom:1.5rem}
div[data-testid="stMetric"] {background:rgba(29,185,84,.08);border:1px solid rgba(29,185,84,.25);padding:1rem;border-radius:.8rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎵 Spotify Track Popularity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate a track’s popularity from its metadata using machine learning.</div>', unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Add best_model.joblib to the same GitHub repository folder as app.py.")
    st.info("In Colab, download it with: files.download('best_model.joblib')")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error("Could not load the model. Check scikit-learn/joblib version compatibility.")
    st.exception(exc)
    st.stop()

classifier = is_classifier(model)

with st.sidebar:
    st.header("🎧 Song details")
    st.caption("Enter the metadata and click Predict.")
    with st.form("prediction_form"):
        song_name = st.text_input("Song name", "Example Song")
        genre = st.text_input("Genre", "pop")
        artist = st.text_input("Artist(s)", "Example Artist")
        album = st.text_input("Album", "Example Album")
        duration_min = st.number_input("Duration (minutes)", min_value=0.1, max_value=30.0, value=3.5, step=0.1)
        explicit = st.selectbox("Explicit content?", [False, True], format_func=lambda x: "Yes" if x else "No")
        submitted = st.form_submit_button("✨ Predict", type="primary", use_container_width=True)

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("How it works")
    st.write("The app sends your inputs to the saved preprocessing-and-model pipeline. It applies the same encoding and transformations learned during training before predicting.")
    st.markdown("**Input features**\n- Song name, genre, artist and album\n- Duration in minutes\n- Explicit-content flag")
    st.warning("Educational estimate only. This app does not query Spotify and cannot guarantee an actual or future Spotify popularity score.")

with right:
    st.subheader("Prediction")
    if submitted:
        if not all([song_name.strip(), genre.strip(), artist.strip(), album.strip()]):
            st.error("Please complete all text fields.")
        else:
            row = pd.DataFrame([{
                "name": song_name.strip(),
                "genre": genre.strip(),
                "artists": artist.strip(),
                "album": album.strip(),
                "duration_min": float(duration_min),
                "explicit": explicit
            }])
            try:
                prediction = model.predict(row)[0]
                if classifier:
                    if int(prediction) == 1:
                        st.success("### Popular")
                        st.metric("Predicted class", "Popular")
                    else:
                        st.info("### Not Popular")
                        st.metric("Predicted class", "Not Popular")
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(row)[0]
                        classes = list(model.classes_)
                        if 1 in classes:
                            p = float(probabilities[classes.index(1)])
                            st.metric("Estimated probability of Popular", f"{p:.1%}")
                            st.progress(max(0, min(100, int(round(p * 100)))))
                else:
                    score = max(0.0, min(100.0, float(prediction)))
                    st.metric("Predicted popularity", f"{score:.1f} / 100")
                    st.progress(int(round(score)))
                    st.caption("This is a model estimate clipped to the dataset’s 0–100 range, not an official Spotify score.")
            except Exception as exc:
                st.error("Prediction failed. Check that the model was trained with the expected feature columns.")
                st.exception(exc)
    else:
        st.info("Fill in the song details in the sidebar and click Predict.")
        st.metric("Predicted result", "—")

st.divider()
st.subheader("About this project")
c1, c2, c3 = st.columns(3)
c1.markdown("**Data preparation**\n\nCleaning, encoding and scaling")
c2.markdown("**Machine learning**\n\nTrained scikit-learn pipeline")
c3.markdown("**Deployment**\n\nStreamlit web application")
st.caption("Student project demo • Predictions depend on the training data and model quality.")
