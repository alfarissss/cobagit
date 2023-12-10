import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("EfficientNet_model2.h5", compile=False)
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

# Tambahkan kamus deskripsi kucing
cat_descriptions = {
    'Abyssinian': "Deskripsi Abyssinian",
    'American Bobtail': "Deskripsi American Bobtail",
    'American Shorthair': "Deskripsi American Shorthair",
    'Bengal': "Deskripsi Bengal",
    'Birman': "Deskripsi Birman",
    'Bombay': "Deskripsi Bombay",
    'British Shorthair': "Deskripsi British Shorthair",
    'Egyptian Mau': "Deskripsi Egyptian Mau",
    'Maine Coon': "Deskripsi Maine Coon",
    'Persian': "Deskripsi Persian",
    'Ragdoll': "Deskripsi Ragdoll",
    'Russian Blue': "Deskripsi Russian Blue",
    'Siamese': "Deskripsi Siamese",
    'Sphynx': "Deskripsi Sphynx",
    'Tuxedo': "Deskripsi Tuxedo"
}


@app.route('/')
def index():
    return jsonify({
        "status" : {
            "code" : 200,
            "message" : "Succes fetching the API",
        },
        "data" : None,
    }),200

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Augmentasi image
            img = tf.keras.preprocessing.image.load_img(image_path)
            img_arr = tf.keras.utils.img_to_array(img)
            img_arr = img_arr / 255.
            img_arr = tf.image.resize(img_arr, [224, 224])
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Prediksi dari model
            predictions = model.predict(img_arr)
            # Label prediksi
            predicted_label = labels[np.argmax(predictions)]
            # Confidence score
            confidence_score = float(predictions[0][np.argmax(predictions)])
            # Ambil nama ras kucing tanpa angka dari label prediksi
            cat_breed_name = predicted_label[2:]
            # Dapatkan deskripsi dari kamus
            cat_breed_description = cat_descriptions.get(cat_breed_name)



            return jsonify({
                "status": {
                    "code": 200,
                    "photo": request.host_url + image_path,
                    "message": "Success Predict!",
                },
                "data": {
                    "Cat_breed_Predictions": cat_breed_name,
                    "Cat_breed_Description": cat_breed_description,
                    "confidence": confidence_score
                }
            }),200
        else:
            return jsonify({
                "status" : {
                    "code" : 400,
                    "message" : "Bad request",
                },
                "data" : None,
            }),400
    else:
        return jsonify({
            "status" : {
                "code" : 405,
                "message" : "Method not allowed",
            },
            "data" : None,
        }),405


if __name__ == "__main__":
    app.run()

    