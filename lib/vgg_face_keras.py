#!/usr/bin python3
""" VGG_Face inference using OpenCV-DNN
Model from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import logging
import sys
import os

import cv2
import numpy as np
from fastcluster import linkage, linkage_vector

from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



from keras.models import Model
from keras.layers import Input, Convolution2D as Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
import keras.backend as K
from keras.utils.data_utils import get_file


def VGG16(include_top=False, weights='vggface',
          input_shape=None,
          pooling=None,
          classes=2622):

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(classes, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_vgg16')  # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5",
                                cache_subdir=".")
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5",
                                cache_subdir=".")
    model.load_weights(weights_path, by_name=True)
    model.layers.pop()
    model.layers.pop()
    return model


class VGGFace():
    """ VGG Face feature extraction.
        Input images should be in BGR Order """

    def __init__(self, backend="GPU"):
        logger.debug("Initializing %s: (backend: %s)", self.__class__.__name__, backend)
        git_model_id = 7
        model_filename = ["vgg_face_v1.caffemodel", "vgg_face_v1.prototxt"]
        self.input_size = 224
        # Average image provided in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
        self.average_img = [129.1863, 104.7624, 93.5940]

        self.model = self.get_model(git_model_id, model_filename, backend)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    def get_model(self, git_model_id, model_filename, backend):
        """ Check if model is available, if not, download and unzip it """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        cache_path = os.path.join(root_path, "plugins", "extract", ".cache")
        #model = GetModel(model_filename, cache_path, git_model_id).model_path
        #model = cv2.dnn.readNetFromCaffe(model[1], model[0])  # pylint: disable=no-member
        #model.setPreferableTarget(self.get_backend(backend))

        model = VGG16(include_top=True, input_shape=(224,224,3), pooling="avg")
        return model

    @staticmethod
    def get_backend(backend):
        """ Return the cv2 DNN backend """
        if backend == "OPENCL":
            logger.info("Using OpenCL backend. If the process runs, you can safely ignore any of "
                        "the failure messages.")
        retval = getattr(cv2.dnn, "DNN_TARGET_{}".format(backend))  # pylint: disable=no-member
        return retval

    def predict(self, face):
        """ Return encodings for given image from vgg_face """
        if face.shape[0] != self.input_size:
            face = self.resize_face(face)
#         blob = cv2.dnn.blobFromImage(face,  # pylint: disable=no-member
#                                      1.0,
#                                      (self.input_size, self.input_size),
#                                      self.average_img,
#                                      False,
#                                      False)
#         self.model.setInput(blob)
#         preds = self.model.forward("fc7")[0, :]
        preds = self.model.predict(np.expand_dims(face, axis=0))[0, :]
        return preds

    def resize_face(self, face):
        """ Resize incoming face to model_input_size """
        if face.shape[0] < self.input_size:
            interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint:disable=no-member

        face = cv2.resize(face,  # pylint:disable=no-member
                          dsize=(self.input_size, self.input_size),
                          interpolation=interpolation)
        return face

    @staticmethod
    def find_cosine_similiarity(source_face, test_face):
        """ Find the cosine similarity between a source face and a test face """
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        return 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))

    def sorted_similarity(self, predictions, method="ward"):
        """ Sort a matrix of predictions by similarity Adapted from:
            https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - predictions is a stacked matrix of vgg_face predictions shape: (x, 4096)
            - method = ["ward","single","average","complete"]
        output:
            - result_order is a list of indices with the order implied by the hierarhical tree

        sorted_similarity transforms a distance matrix into a sorted distance matrix according to
        the order implied by the hierarchical tree (dendrogram)
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        num_predictions = predictions.shape[0]
        try:
            result_linkage = linkage(predictions, method=method, preserve_input=False)
        except MemoryError as error:
            logger.info("Ran out of memory with sort, switching to slower sort.")
            result_linkage = linkage_vector(predictions, method=method)
        result_order = self.seriation(result_linkage,
                                      num_predictions,
                                      num_predictions + num_predictions - 2)

        return result_order

    def seriation(self, tree, points, current_index):
        """ Seriation method for sorted similarity
            input:
                - tree is a hierarchical tree (dendrogram)
                - points is the number of points given to the clustering process
                - current_index is the position in the tree for the recursive traversal
            output:
                - order implied by the hierarchical tree

            seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if current_index < points:
            return [current_index]
        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])
        return self.seriation(tree, points, left) + self.seriation(tree, points, right)
