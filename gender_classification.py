import numpy as np
import cv2
from keras.models import Model
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D

_base_dir = os.path.dirname(__file__)

def _resize_without_distortion(image, to_h, to_w):
    h, w = image.shape[:2]
    mean = image.mean(axis=(0, 1))
    input_aspect_ratio = w / h
    target_aspect_ratio = to_w / to_h
    # making the aspect ratio of the input image the same as the target by padding it
    pad_top = pad_bottom = pad_left = pad_right = 0
    if input_aspect_ratio < target_aspect_ratio:
        d = int((h * target_aspect_ratio) - w)
        pad_left = d // 2
        pad_right = d - pad_left
    else:
        d = int((w / target_aspect_ratio) - h)
        pad_top = d // 2
        pad_bottom = d - pad_top


    pad = ((pad_top, pad_bottom), (pad_left, pad_right))
    if image.ndim == 3:
        pad = pad + ((0, 0),)

    padded_image = np.pad(image, pad, mode='constant', constant_values=[(v, v) for v in mean])
    return cv2.resize(padded_image, (to_w, to_h))

class FaceExtractor():
    def __init__(self,
                 prototxt_path = os.path.join(_base_dir, 'face_bbox_model_data/deploy.prototxt'),
                 caffemodel_path=os.path.join(_base_dir, 'face_bbox_model_data/weights.caffemodel')
                 ):

        self.model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def extract_face(self, image, h_pad=0.5, w_pad=0.5, confidence_threshold=0.5):
        blob = cv2.dnn.blobFromImage(_resize_without_distortion(image, 300, 300),
                                     1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()
        detections = detections.squeeze()
        # assumes single face in the image
        max_confidence_idx = np.argmax(detections[:, 2])
        confidence = detections[max_confidence_idx, 2]
        # if we can't find any bbox with high condidence
        if confidence < confidence_threshold:
            return image

        #cropping the face
        (h, w) = image.shape[:2]
        box = detections[max_confidence_idx, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        box_h = (endY - startY)
        box_w = (endX - startX)

        startY = max(startY - int(h_pad * box_h), 0)
        endY = min(endY + int(h_pad * box_h), h)

        startX = max(startX - int(w_pad * box_w), 0)
        endX = min(endX + int(w_pad * box_w), w)

        return image[startY:endY, startX:endX]


class GenderClassifier:
    FEMALE_CLASS = 0
    MALE_CLASS = 1

    CLASS_NAMES = {FEMALE_CLASS:"Female", MALE_CLASS:"Male"}
    def __init__(self, weights_path = os.path.join(_base_dir, './gender_nn_weights/weights.best.inc.male.hdf5')):
        self._model = self._build_model()
        self._model_input_dim= (218, 178) # This are the dimension used to training
        self._model.load_weights(weights_path)
        self._faceExtractor = FaceExtractor()

    def _build_model(self):
        inc_model = InceptionV3(weights=None,
                                include_top=False,
                                input_shape=(None, None, 3)
                                )

        x = inc_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        return Model(inputs=inc_model.input, outputs=predictions)



    def classify_image(self, image, female_confidence_threshold=0.4,
                       return_confidence=False, crop_face=False):
        '''image can be a numpy array (the result of imread) or path to an image'''
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        if crop_face:
            image = self._faceExtractor.extract_face(image)

        image = _resize_without_distortion(image, *self._model_input_dim)
        image = preprocess_input(image)

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        pred = self._model.predict(image)[0]

        res_class = GenderClassifier.MALE_CLASS
        if pred[GenderClassifier.FEMALE_CLASS] >= female_confidence_threshold:
            res_class = GenderClassifier.FEMALE_CLASS

        if return_confidence:
            return GenderClassifier.CLASS_NAMES[res_class], pred[res_class]

        return GenderClassifier.CLASS_NAMES[res_class]
