from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense


class Network(Model):
    def __init__(self, frame_shape, window_legth, actions_num):
        self.inputs_shape = (window_legth, frame_shape[0], frame_shape[1])
        self.actions_num = actions_num

    def create_model(self):
        frame = Input(shape=(self.inputs_shape))
        cv1 = Convolution2D(
            32,
            kernel_size=(8, 8),
            strides=4,
            activation="relu",
            data_format="channels_first",
        )(frame)
        cv2 = Convolution2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="relu",
            data_format="channels_first",
        )(cv1)
        cv3 = Convolution2D(
            64,
            kernel_size=(3, 3),
            strides=1,
            activation="relu",
            data_format="channels_first",
        )(cv2)
        dense = Flatten()(cv3)
        dense = Dense(512, activation="relu")(dense)
        buttons = Dense(self.actions_num, activation="linear")(dense)
        return Model(inputs=frame, outputs=buttons)
