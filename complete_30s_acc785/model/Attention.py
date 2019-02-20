# -*- coding: utf-8 -*-
# @Time    : 2019/2/5 23:29
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers,regularizers,constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # 后面的层不需要mask了，所以这里可以直接返回none
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        # 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
        step_dim = self.step_dim

        # 输入和参数分别reshape再点乘后，tensor.shape变成了(batch_size*timesteps, 1),之后每个batch要分开进行归一化
		# 所以应该有 eij = K.reshape(..., (-1, timesteps))
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        # RNN一般默认激活函数为tanh, 对attention来说激活函数差别不打，因为要做softmax
        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
        # 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们的attention权重设为0
            a *= K.cast(mask, K.floatx())
        # cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。
		# 比如shape = (3,)变成 (3, 1)
        a = K.expand_dims(a)
        # 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)
        weighted_input = x * a

        # weighted_input的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
		# weighted_input在axis=1上取和，返回值的shape为 (batch_size, 1, units)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # 返回的结果是c，其shape为 (batch_size, units)
        return input_shape[0],  self.features_dim
