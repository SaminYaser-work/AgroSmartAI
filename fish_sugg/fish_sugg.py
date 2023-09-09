import numpy as np
from joblib import load

labels = ['Catla',
          'Sing',
          'Prawn',
          'Rui',
          'Koi',
          'Pangas',
          'Tilapia',
          'Silver Carp',
          'Karpio',
          'Magur',
          'Shrimp']


def predict_fish(data):
    model = load('./models/tpot_fish2.joblib')
    pred = model.predict_proba([data])
    pred_fish = labels[np.argmax(pred, axis=1)[0]]
    pred_conf = np.max(pred, axis=1)[0]
    return {
        'fish': pred_fish,
        'confidence': pred_conf
    }
