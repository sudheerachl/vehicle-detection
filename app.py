from flask import Flask, request, redirect, url_for, render_template
import json
from myfile import process_image as display_image_info  # Import your model code
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configure upload folder (adjust path if needed)
UPLOAD_FOLDER = os.path.join(app.instance_path, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

# Set empty build command (assuming no build process needed)
app.config['FLASK_ENV'] = 'development'  # Optional (may not be necessary)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return redirect(request.url)  # Redirect if no image uploaded

        image = request.files['imagefile']
        if image.filename == '':
            return redirect(request.url)  # Redirect if empty filename

        # Securely generate filename
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            # Save the image
            image.save(image_path)

            # Run the model on the saved image
            results = display_image_info(image_path)

            # Convert results to JSON format (assuming results is a dictionary or list)
            prediction = json.dumps(results, indent=4)

            # Return JSON data for further processing (uncomment if preferred)
            # return prediction

            # Render template with prediction (default behavior)
            return render_template('index.html', prediction=prediction)
        except Exception as e:  # Catch any errors during processing
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)

    return render_template('index.html')  # Render template for GET requests

if __name__ == '__main__':
    app.run()  # Remove debug=True for deployment
