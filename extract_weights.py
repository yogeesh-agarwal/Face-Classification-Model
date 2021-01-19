import os
import pickle
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

latest_ckp = tf.train.latest_checkpoint("./saved_models")
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
var_to_shape_map = reader.get_variable_to_shape_map()

model_vars = {}
for key in var_to_shape_map:
    model_vars[key] = reader.get_tensor(key)

for key in sorted(model_vars):
    print(key , model_vars[key].shape)

conv_weights = {}
bn_weights = {}

for i in range(19):
    conv_layer_name = "Variable" if i == 0 else "Variable_{}".format(i)
    conv_weights["conv_{}".format(i)] = model_vars[conv_layer_name]

    bn_vars = {"beta" : "beta" if i == 0 else "beta_{}".format(i) , "gamma" : "gamma" if i == 0 else "gamma_{}".format(i) , "mean" : "moments/Squeeze/ExponentialMovingAverage" if i == 0 else "moments_{}/Squeeze/ExponentialMovingAverage".format(i) , "variance" : "moments/Squeeze_1/ExponentialMovingAverage" if i == 0 else "moments_{}/Squeeze_1/ExponentialMovingAverage".format(i)}
    bn_weight = {}
    for key in bn_vars:
        bn_weight[key] = model_vars[bn_vars[key]]
    bn_weights["conv_{}".format(i)] = bn_weight

with open("./data/conv_weights.pickle" , "wb") as content:
    pickle.dump(conv_weights, content , protocol = pickle.HIGHEST_PROTOCOL)
with open("./data/bn_weights.pickle" , "wb") as content:
    pickle.dump(bn_weights , content , protocol = pickle.HIGHEST_PROTOCOL)

print("conv and bn weights are extracted and stored in pickle file")
