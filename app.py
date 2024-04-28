from flask import Flask, request, redirect, url_for, render_template
import json
from myfile import process_image as display_image_info
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set empty build command (assuming no build process needed)
app.config['FLASK_ENV'] = 'development'  # Optional (may not be necessary)
app.debug = True 




@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return redirect(request.url)  # Redirect if no image uploaded

        image = request.files['imagefile']
        if image.filename == '':
            return redirect(request.url)  # Redirect if empty filename

        # Securely generate filename (optional, not used here)
        # filename = secure_filename(image.filename)

        try:
            # Process the image directly using a file-like object
            results = display_image_info(image)  # Pass the image object itself

            # Convert results to JSON format (assuming results is a dictionary or list)
            prediction = json.dumps(results, indent=4)

            # Render template with prediction (default behavior)
            return render_template('index.html', prediction=prediction)
        except Exception as e:  # Catch any errors during processing
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)

    return render_template('index.html')  # Render template for GET requests


if __name__ == '__main__':
    app.run(debug=False)  # Remove debug=True for deployment
