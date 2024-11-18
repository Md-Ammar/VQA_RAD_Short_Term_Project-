from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import ViltProcessor, ViltForQuestionAnswering
import os

# Initialize Flask app
app = Flask(__name__)

# Set up a folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the fine-tuned model and processor
def load_model():
    path = "path_to_your_model"  # Replace with the correct model path
    processor = ViltProcessor.from_pretrained(path)
    model = ViltForQuestionAnswering.from_pretrained(path)
    return processor, model

processor, model = load_model()

# Home route to render the web page
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    answer = None

    if request.method == "POST":
        # Handle the uploaded image and question
        if 'image' not in request.files or request.form['question'] == '':
            return redirect(request.url)

        file = request.files['image']
        question = request.form['question']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and process the image
            image = Image.open(filepath)

            # Prepare the input for the model
            encoding = processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)

            # Get the model prediction
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits

                probs = F.softmax(logits, dim=-1)
                idx = torch.argmax(probs, dim=-1).item()

            answer = model.config.id2label[idx]  # Get the predicted label

            return render_template("index.html", uploaded_image=url_for('static', filename=f'uploads/{filename}'), answer=answer)

    return render_template("index.html", uploaded_image=None, answer=None)

if __name__ == "__main__":
    app.run(debug=True)
