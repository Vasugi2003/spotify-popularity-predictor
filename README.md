# Spotify Track Popularity Predictor

A student machine-learning project deployed as a Streamlit web app. It loads a trained scikit-learn model pipeline from `best_model.joblib` and predicts track popularity from metadata.

## Files required
- `app.py` — Streamlit user interface
- `best_model.joblib` — your trained model pipeline
- `requirements.txt` — required Python packages

## Add your model
Upload your own `best_model.joblib` into the repository root, in the same folder as `app.py`.

The model should be the complete scikit-learn pipeline including preprocessing and the trained estimator. For regression, save with `joblib.dump(best_model, "best_model.joblib")`. For classification, save the classifier pipeline, e.g. `joblib.dump(best_classifier, "best_model.joblib")`.

The expected input columns are `name`, `genre`, `artists`, `album`, `duration_min`, and `explicit`.

## Free deployment with Streamlit Community Cloud
1. Create a GitHub repository (public is simplest for a student portfolio).
2. Upload `app.py`, `requirements.txt`, and `best_model.joblib` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt` and gives you a shareable URL.

## Test locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting
- **Model file not found:** put `best_model.joblib` in the same folder as `app.py`.
- **Version errors:** use compatible scikit-learn/joblib versions to the ones used to train and save the model.
- **Feature-name errors:** ensure the model pipeline was trained with the six expected columns.
- **Large model:** GitHub has file-size limits; if your model is too large, use external model storage.

## Resume bullet
Built and deployed a Spotify track popularity prediction web application using Python, scikit-learn, and Streamlit; implemented an interactive UI and model inference pipeline and hosted the application using Streamlit Community Cloud.

## Disclaimer
This is an educational prediction based on the training dataset. It is not an official Spotify service and does not retrieve live Spotify scores.
