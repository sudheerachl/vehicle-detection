from flask import Flask, request, redirect, url_for, render_template
import json
from myfile import display_image_info  # Import your model code
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = '/images'  # Define upload folder for images
@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return redirect(request.url)  # Redirect back if no image uploaded

        image = request.files['imagefile']
        if image.filename == '':
            return redirect(request.url)  # Redirect back if empty filename

        # Save the image
        filename = secure_filename(image.filename)
        image_path = "./images/" + filename
        image.save(image_path)

        # Run the model on the saved image
        results = display_image_info(image_path)

        # Convert results to JSON format
        json_output = json.dumps(results, indent=4)  # Assuming results is a dictionary or list

        # You can choose to display results in the template or return JSON for further processing
        # return json_output  # Uncomment to return JSON directly
        return render_template('index.html', prediction=json_output)

    return render_template('index.html')  # Render template for GET requests

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=True)
