from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

# Load the model
model_path = 'apple_classifier_resnet502.h5'  # Update with your model's path
model = load_model(model_path)

class_labels = ['Apple_Bad', 'Apple_Good']

# Flag to control live camera feed
camera_running = False

def predict(image):
    """Predict whether the apple is good or bad."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Update size based on your model's input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    model_class = np.argmax(prediction[0])
    model_label = class_labels[model_class]
    fruit_name = model_label.split("_")[0]
    fruit_quality = model_label.split("_")[1]
    
    return fruit_name, fruit_quality

# ... Remaining code remains the same ...


def detect_apple(image):
    """Detect the apple in the image and return the bounding box."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h
    return None

def upload_image():
    """Upload an image for classification and drawing box."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    fruit_name, fruit_quality = predict(img)

    # Detect apple and draw bounding box
    bbox = detect_apple(img)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    # Display the image and result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    label_img.config(image=img_tk)
    label_img.image = img_tk
    label_result.config(text=f"Result: {fruit_name} ({fruit_quality})")

def live_detection():
    """Use camera for live detection."""
    global camera_running
    camera_running = True
    cap = cv2.VideoCapture(0)
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        fruit_name, fruit_quality = predict(frame)

        # Detect apple and draw bounding box
        bbox = detect_apple(frame)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

        cv2.putText(frame, f'{fruit_name} ({fruit_quality})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_live_detection():
    """Stop the live detection feed."""
    global camera_running
    camera_running = False

def on_closing():
    """Ensure the program closes properly when the window is closed."""
    stop_live_detection()
    root.quit()

def close_app():
    """Close the application when the Close button is clicked."""
    stop_live_detection()
    root.quit()

# Create GUI
root = tk.Tk()
root.title("Apple Quality Detector")
root.geometry("500x500")
root.config(bg="#f0f0f0")  # Light gray background

# Set window icon
# root.iconbitmap('icon.ico')  # You can add an .ico file here for a custom icon

# Header Label
header_label = tk.Label(root, text="Apple Quality Detector", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#4CAF50")
header_label.pack(pady=20)

# Frame for Buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

btn_upload = tk.Button(button_frame, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#4CAF50", fg="white", relief="raised", width=20)
btn_upload.grid(row=0, column=0, padx=10)

btn_live = tk.Button(button_frame, text="Live Detection", command=lambda: threading.Thread(target=live_detection, daemon=True).start(), font=("Arial", 14), bg="#4CAF50", fg="white", relief="raised", width=20)
btn_live.grid(row=0, column=1, padx=10)

btn_close = tk.Button(button_frame, text="Close", command=close_app, font=("Arial", 14), bg="#f44336", fg="white", relief="raised", width=20)
btn_close.grid(row=1, column=0, columnspan=2, pady=20)

# Frame for displaying image and result
image_frame = tk.Frame(root, bg="#f0f0f0")
image_frame.pack(pady=10)

label_img = tk.Label(image_frame, bg="#f0f0f0")
label_img.pack()

label_result = tk.Label(root, text="Result: ", font=("Arial", 16), bg="#f0f0f0", fg="#4CAF50")
label_result.pack(pady=10)

# Bind the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()