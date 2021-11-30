from PIL import Image
import numpy as np
import joblib
import json
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def loadImage(im_path):
    im = Image.open(im_path).convert('L')

    return im

def preProcessing(im):
    if im.size != (28,28):
        im = im.resize((28,28))

    im = np.asarray(im).reshape((1,784))

    return im

# Image Path
INPUTS           = os.listdir('inputs/')
SAVED_CLASSIFIER = 'MNIST_classifier_MLP.joblib'
IM_PATH          = ['inputs/' + im for im in INPUTS]

# Pipeline
im_loader = FunctionTransformer(loadImage)
scaling   = FunctionTransformer(preProcessing)
estimator = joblib.load(SAVED_CLASSIFIER)

model = make_pipeline(im_loader, scaling, estimator)

# Process and Save
N           = len(INPUTS)
predictions = dict(zip([INPUTS[k] for k in range(N)], [model.predict(IM_PATH[k])[0] for k in range(N)]))

if predictions:
    with open('output/digit_prediction.json', 'w') as output:
        json.dump(predictions, output)
        print('done')

        
