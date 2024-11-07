from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import backend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'

# Create the upload directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Serve the HTML file
@app.route('/')
def index():
    return render_template('b.html')

# Endpoint for video upload and deepfake detection
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Call the detection function from `backend.py`
        is_deepfake = backend.detect_deepfake(filepath)
        response = jsonify({'deepfake': is_deepfake})
    except Exception as e:
        print(f"Error during detection: {e}")
        response = jsonify({'error': 'Deepfake detection failed'}), 500
    finally:
        # Clean up the file after processing
        os.remove(filepath)
    
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
