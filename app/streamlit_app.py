import joblib
import streamlit as st

MODEL_PATH = "models/tfidf_lr.joblib"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def main():
    st.title("Hotel Review Fake Detector")
    st.write("Enter a review to check if it's likely deceptive (fake) or truthful.")

    model = None
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.warning(f"Model not found at {MODEL_PATH}. Train the model first: python src/train.py")

    text = st.text_area("Review text", height=200)
    if st.button("Predict") and text.strip():
        if model is None:
            st.error("No model available")
            return
        pred = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        label = "Deceptive / Fake" if int(pred) == 1 else "Truthful / Real"
        st.write("**Prediction:**", label)
        st.write("**Confidence:**")
        st.write(f"Class 0 (truthful): {probs[0]:.3f}")
        st.write(f"Class 1 (deceptive): {probs[1]:.3f}")


if __name__ == "__main__":
    main()
