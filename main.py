import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier App")

        self.image_path = None

        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Helvetica', 12))
        self.select_button = ttk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        self.run_button = ttk.Button(master, text="Run Classifier", command=self.run_classifier)
        self.run_button.grid(row=0, column=1, pady=10, padx=10, sticky="e")

        self.result_label = ttk.Label(master, text="", font=("Helvetica", 12))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10, padx=10)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.png;*.jpeg")])

        if self.image_path:
            image_obj = Image.open(self.image_path)


    def run_classifier(self):
        if self.image_path:
            model = InceptionV3(weights='imagenet')
            predictions = self.classify_image(model, self.image_path)

            result_text = ""
            for i, (imagenet_id, label, score) in enumerate(predictions):
                result_text += f"{i + 1}: {label} ({score * 100:.2f}%)\n"

            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Please select an image first.")

    def classify_image(self, model, image_path):
        img_array = self.load_and_preprocess_image(image_path)
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions)
        return decoded_predictions[0]

    def load_and_preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array


root = tk.Tk()
app = ImageClassifierApp(root)
root.mainloop()