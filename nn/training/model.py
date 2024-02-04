import tfquaternion as tfq
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras import backend as K

def quat_mult_error(y_true, y_pred):
    q = tfq.Quaternion(y_pred).normalized()
    q_hat = tfq.quaternion_conjugate(y_true)
    q_prod = q * q_hat
    q_prod = tf.convert_to_tensor(q_prod)
    w, x, y, z = tf.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1)))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
    #def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        #for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        #loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        # return K.concatenate(inputs, -1)
        return self.multi_loss(ys_true, ys_pred)


def create_pred_model_6d_quat(window_size=200, mag=False):
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')

    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    convC1 = Conv1D(128, 11)(x3)
    convC2 = Conv1D(128, 11)(convC1)
    poolC = MaxPooling1D(3)(convC2)

    if mag:
        AB = concatenate([poolA, poolB, poolC])
    else:
        AB = concatenate([poolA, poolB])
    
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(LSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    model = Model([x1, x2, x3], [y1_pred, y2_pred])

    model.summary()
    
    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')
    y1_pred, y2_pred = pred_model([x1, x2, x3])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    train_model = Model([x1, x2, x3, y1_true, y2_true], out)
    train_model.summary()
    return train_model