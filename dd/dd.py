import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model_path = './models/ac.tflite'

labels = {
    "Pepper Bell Bacterial Spot": "A bacterial disease caused by Xanthomonas campestris pv. vesicatoria. It is the most common foliar disease of peppers and can cause early defoliation, yield losses, and poor fruit quality. The disease is characterized by the presence of small, water-soaked spots on leaves that eventually turn brown and necrotic. The spots may be angular or irregular in shape and may have a yellow halo. The disease can also affect stems and fruit, causing small, scabby lesions.",
    "Pepper Bell Healthy": "A healthy pepper bell plant is typically green and vigorous, with dark green leaves and few or no blemishes. The fruit should be firm and bright red, with no signs of disease or pests.",
    "Potato Early Blight": "A fungal disease caused by Alternaria solani. It is one of the most common diseases of potatoes and can cause significant yield losses. The disease is characterized by the presence of dark brown, water-soaked spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and tubers, causing black, sunken lesions.",
    "Potato Late Blight": "A fungal disease caused by Phytophthora infestans. It is a more serious disease than early blight and can cause complete crop loss. The disease is characterized by the presence of large, dark brown, water-soaked spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and tubers, causing black, sunken lesions.",
    "Potato Healthy": "A healthy potato plant is typically green and vigorous, with dark green leaves and few or no blemishes. The tubers should be firm and smooth, with no signs of disease or pests.",
    "Tomato Bacterial Spot": "A bacterial disease caused by Xanthomonas campestris pv. vesicatoria. It is the most common foliar disease of tomatoes and can cause early defoliation, yield losses, and poor fruit quality. The disease is characterized by the presence of small, water-soaked spots on leaves that eventually turn brown and necrotic. The spots may be angular or irregular in shape and may have a yellow halo. The disease can also affect stems and fruit, causing small, scabby lesions.",
    "Tomato Early Blight": "A fungal disease caused by Alternaria solani. It is one of the most common diseases of tomatoes and can cause significant yield losses. The disease is characterized by the presence of dark brown, water-soaked spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and fruit, causing black, sunken lesions.",
    "Tomato Late Blight": "A fungal disease caused by Phytophthora infestans. It is a more serious disease than early blight and can cause complete crop loss. The disease is characterized by the presence of large, dark brown, water-soaked spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and fruit, causing black, sunken lesions.",
    "Tomato Leaf Mold": "A fungal disease caused by Septoria lycopersici. It is a common disease of tomatoes and can cause significant yield losses. The disease is characterized by the presence of small, brown spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and fruit, causing small, scabby lesions.",
    "Tomato Septoria Leaf Spot": "A fungal disease caused by Septoria lycopersici. It is a common disease of tomatoes and can cause significant yield losses. The disease is characterized by the presence of small, brown spots on leaves. The spots eventually enlarge and coalesce, causing the leaves to wilt and die. The disease can also affect stems and fruit, causing small, scabby lesions.",
    "Tomato Spider Mites": "Tiny, eight-legged pests that feed on the undersides of leaves. They can cause significant damage to tomatoes, causing the leaves to turn yellow and wilt. The mites are difficult to see with the naked eye, but they can be detected by the presence of fine, webbing on the leaves.",
    "Tomato Target Spot": "A fungal disease caused by Corynespora cassiicola. It is a common disease of tomatoes and can cause significant yield losses. The disease is characterized by the presence of small, brown spots on leaves. The spots have a yellow halo and a target-like appearance. The disease can also affect stems and fruit, causing small, scabby lesions.",
    'Tomato Tomato Yellow Leaf Curl Virus': 'A virus that causes yellowing and curling of leaves in tomatoes. The virus is spread by aphids and can be very difficult to control. There is no cure for tomato yellow leaf curl virus, but there are some cultural practices that can help to reduce the spread of the virus, such as planting resistant varieties and avoiding planting tomatoes near crops that are known to be hosts of the virus.',
    'Tomato Tomato Mosaic Virus': 'A virus that causes a mosaic pattern on the leaves of tomatoes. The virus is spread by aphids and can also be spread by contact with infected plants. There is no cure for tomato mosaic virus, but there are some cultural practices that can help to reduce the spread of the virus, such as planting resistant varieties and avoiding planting tomatoes near crops that are known to be hosts of the virus.',
    'Tomato Healthy': 'A healthy tomato plant is typically green and vigorous, with dark green leaves and few or no blemishes. The fruit should be firm and bright red, with no signs of disease or pests.'
}


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
            {
                "name": list(labels.keys())[i],
                "confidence": round(res[0][i] * 100, 2),
                "description": list(labels.values())[i]
            }
        )
    conf.sort(key=lambda x: x["confidence"], reverse=True)
    return conf
