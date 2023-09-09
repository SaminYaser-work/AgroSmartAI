import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model_path = './models/ac.tflite'

labels = ['Pepper Bell Bacterial Spot',
          'Pepper Bell Healthy',
          'Potato Early Blight',
          'Potato Late Blight',
          'Potato Healthy',
          'Tomato Bacterial Spot',
          'Tomato Early Blight',
          'Tomato Late Blight',
          'Tomato Leaf Mold',
          'Tomato Septoria Leaf Spot',
          'Tomato Spider Mites',
          'Tomato Target Spot',
          'Tomato Tomato Yellow Leaf Curl Virus',
          'Tomato Tomato Mosaic Virus',
          'Tomato Healthy']


def predict_diseases(data):
    test_img = image.load_img(data, target_size=(48, 48))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], test_img)

    interpreter.invoke()

    res = interpreter.get_tensor(output_details[0]['index'])

    conf = []
    for i in range(len(res[0])):
        conf.append(
            {"name": labels[i], "confidence": round(res[0][i] * 100, 2)}
        )
    conf.sort(key=lambda x: x["confidence"], reverse=True)
    return conf
