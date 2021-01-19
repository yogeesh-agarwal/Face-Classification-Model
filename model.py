import tensorflow as tf

class Darknet19:
    def __init__(self , num_classes , is_training):
        self.num_classes = num_classes
        self.is_training = is_training

    def batch_norm(self , inputs, n_out):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        return normed

    def init_weights(self , shape):
        weights = tf.truncated_normal(shape , stddev = 0.1)
        return tf.Variable(weights)

    def conv_block(self, input , filter_shape , strides = 1 , padding = "SAME" , if_mp = False):
        conv_weights = self.init_weights(filter_shape)
        out_channels = filter_shape[-1]
        strides = [1,strides,strides,1]

        conv_layer = tf.nn.conv2d(input , conv_weights , strides = strides , padding = padding)
        bn_layer = self.batch_norm(conv_layer , out_channels)
        leaky_relu_layer = tf.nn.leaky_relu(bn_layer , alpha = 0.1)
        if if_mp:
            mp_layer =  tf.nn.max_pool(leaky_relu_layer , ksize = [1,2,2,1] , strides=  [1,2,2,1] , padding = "VALID")
            print(mp_layer.shape)
            return mp_layer

        print(leaky_relu_layer.shape)
        return leaky_relu_layer

    def gen_model(self , input):
        layer_1  = self.conv_block(input , [3,3,3,32] , if_mp = True)
        layer_2  = self.conv_block(layer_1 , [3,3,32,64] , if_mp = True)
        layer_3  = self.conv_block(layer_2 , [3,3,64,128])
        layer_4  = self.conv_block(layer_3 , [1,1,128,64])
        layer_5  = self.conv_block(layer_4 , [3,3,64,128] , if_mp = True)
        layer_6  = self.conv_block(layer_5 , [3,3,128,256])
        layer_7  = self.conv_block(layer_6 , [1,1,256,128])
        layer_8  = self.conv_block(layer_7 , [3,3,128,256] , if_mp = True)
        layer_9  = self.conv_block(layer_8 , [3,3,256,512])
        layer_10 = self.conv_block(layer_9 , [1,1,512,256])
        layer_11 = self.conv_block(layer_10 , [3,3,256,512])
        layer_12 = self.conv_block(layer_11 , [1,1,512,256])
        layer_13 = self.conv_block(layer_12 , [3,3,256,512] , if_mp = True)
        layer_14 = self.conv_block(layer_13 , [3,3,512,1024])
        layer_15 = self.conv_block(layer_14 , [1,1,1024,512])
        layer_16 = self.conv_block(layer_15 , [3,3,512,1024])
        layer_17 = self.conv_block(layer_16 , [1,1,1024,512])
        layer_18 = self.conv_block(layer_17 , [3,3,512,1024])
        layer_19 = self.conv_block(layer_18 , [1,1,1024,self.num_classes])

        avg_pool = tf.nn.avg_pool(layer_19 , ksize = [1,7,7,1] , strides = [1,1,1,1] , padding = "VALID")
        softmax_layer = tf.nn.softmax(avg_pool , axis = -1)
        out_layer = tf.reshape(softmax_layer , [tf.shape(softmax_layer)[0] , self.num_classes])
        return out_layer
