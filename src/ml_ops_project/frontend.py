import os
import streamlit as st
import requests
from PIL import Image
import io

def get_backend_url():
    """Get backend URL from environment or use localhost."""
    return os.environ.get("BACKEND", "http://localhost:8001")

def classify_image(image_bytes, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(predict_url, files=files, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it's running!")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

def main():
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()

    st.title("AI vs Real Art Classifier")
    st.write("Upload an artwork to see if it's AI-generated or real!")
    st.write(f"Backend: {backend}")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify button
        if st.button("Classify"):
            with st.spinner("Analyzing artwork..."):
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                result = classify_image(image_bytes, backend=backend)

                if result is not None:
                    predicted_class = result["predicted_class"]
                    probabilities = result["probabilities"]

                    # Display results
                    if predicted_class == "AI":
                        st.error(f"**Prediction: {predicted_class} Generated**")
                    else:
                        st.success(f"**Prediction: {predicted_class} Art**")

                    # Show probabilities
                    st.write("### Confidence Scores:")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("AI Generated", f"{probabilities['AI']*100:.1f}%")
                    with col2:
                        st.metric("Real Art", f"{probabilities['Real']*100:.1f}%")

                    # Bar chart
                    st.bar_chart(probabilities)

if __name__ == "__main__":
    main()
