from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Load the custom-trained YOLOv8 model
MODEL_PATH = "best.pt"  # Replace with your trained model's path
model = YOLO(MODEL_PATH)

# Define your 5 classes
CLASSES = ["Helmet", "Goggles", "Jacket", "Gloves", "Footwear"]

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    detected_classes = set()
    is_image = file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    if is_image:
        # Process image
        results = model(filepath)

        # Save the annotated image
        annotated_image = results[0].plot(line_width=2)
        processed_filename = f"processed_{file.filename}"
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_filepath, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Gather detected classes
        for result in results:
            for cls_id in result.boxes.cls:
                detected_classes.add(CLASSES[int(cls_id)])

    else:
        # Process video
        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        processed_filename = f"processed_{file.filename.rsplit('.', 1)[0]}.mp4"
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        out = cv2.VideoWriter(processed_filepath, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot(line_width=2)
            for result in results:
                for cls_id in result.boxes.cls:
                    detected_classes.add(CLASSES[int(cls_id)])
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()

    # Check for missing classes
    missing_classes = [cls for cls in CLASSES if cls not in detected_classes]

    # Create response
    response = {
        "detected_classes": list(detected_classes),
        "missing_classes": missing_classes,
        "processed_file_path": processed_filepath,
        "is_image": is_image
    }

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT
    app.run(host="0.0.0.0", port=port, debug=False)
