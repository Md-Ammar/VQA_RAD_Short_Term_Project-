import streamlit as st
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import ViltProcessor, ViltForQuestionAnswering

# Load the fine-tuned model and processor
@st.cache_resource
def load_model():
    processor = ViltProcessor.from_pretrained("./vilt-vqa-rad-finetuned-AIC")
    model = ViltForQuestionAnswering.from_pretrained("./vilt-vqa-rad-finetuned-AIC")
    return processor, model

processor, model = load_model()

# Streamlit app
st.title("VQA RAD - Visual Question Answering on Radiology Images")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Enter a question
question = st.text_input("Enter your question about the image:")

if st.button("Get Answer"):
    if uploaded_image is not None and question:
        # Preprocess the image and the question
        encoding = processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)

        # Run the model to get the answer
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits

            # Apply softmax to logits to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Get the index of the highest probability
            idx = torch.argmax(probs, dim=-1).item()

        # Ensure label mapping is correct (assuming model.config.id2label is available)
        answer = model.config.id2label[idx]
        st.write(f"Predicted answer: **{answer}**")
    else:
        st.write("Please upload an image and enter a question.")
