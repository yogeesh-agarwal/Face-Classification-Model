import os
import cv2
import utils
import pickle
import random
import numpy as np
import tensorflow as tf

class Darknet19:
    def  __init__(self , num_classes , conv_weights_file , bn_weights_file):
        self.num_classes = num_classes
        self.conv_weights_file = conv_weights_file
        self.bn_weights_file = bn_weights_file
        self.conv_weights = self.load_pickle(self.conv_weights_file)
        self.bn_weights = self.load_pickle(self.bn_weights_file)
        self.conv_index = 0
        self.bn_epsilon = 1e-05

    def load_pickle(self , file):
        with open(file , "rb") as content:
            return pickle.load(content)

    def conv_block(self , input , strides = 1 , padding = "SAME" , if_mp = False):
        layer_name = "conv_{}".format(self.conv_index)
        self.conv_index += 1
        strides = [1 , strides , strides , 1]

        conv_weights = self.conv_weights[layer_name]
        bn_weights = self.bn_weights[layer_name]
        mean = bn_weights["mean"]
        variance = bn_weights["variance"]
        beta = bn_weights["beta"]
        gamma = bn_weights["gamma"]

        conv_layer = tf.nn.conv2d(input , conv_weights , strides = strides , padding = padding)
        bn_layer = tf.nn.batch_normalization(conv_layer , mean , variance , beta , gamma , self.bn_epsilon)
        leaky_relu_layer = tf.nn.leaky_relu(bn_layer , alpha = 0.1)
        if if_mp:
            mp_layer = tf.nn.max_pool(leaky_relu_layer,  ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
            print(mp_layer.shape)
            return mp_layer

        print(leaky_relu_layer.shape)
        return leaky_relu_layer

    def gen_model(self, input):
        layer_1  = self.conv_block(input , if_mp = True)
        layer_2  = self.conv_block(layer_1 , if_mp = True)
        layer_3  = self.conv_block(layer_2)
        layer_4  = self.conv_block(layer_3)
        layer_5  = self.conv_block(layer_4 , if_mp = True)
        layer_6  = self.conv_block(layer_5)
        layer_7  = self.conv_block(layer_6)
        layer_8  = self.conv_block(layer_7 , if_mp = True)
        layer_9  = self.conv_block(layer_8)
        layer_10 = self.conv_block(layer_9)
        layer_11 = self.conv_block(layer_10)
        layer_12 = self.conv_block(layer_11)
        layer_13 = self.conv_block(layer_12 , if_mp = True)
        layer_14 = self.conv_block(layer_13)
        layer_15 = self.conv_block(layer_14)
        layer_16 = self.conv_block(layer_15)
        layer_17 = self.conv_block(layer_16)
        layer_18 = self.conv_block(layer_17)
        layer_19 = self.conv_block(layer_18)

        avg_pool = tf.nn.avg_pool(layer_19 , ksize = [1,7,7,1] , strides = [1,1,1,1] , padding = "VALID")
        softmax_layer = tf.nn.softmax(avg_pool , axis = -1)
        out_layer = tf.reshape(softmax_layer , [tf.shape(softmax_layer)[0] , self.num_classes])
        return out_layer


def load_pickle(file):
    with open(file , "rb") as content:
        return pickle.load(content)

def get_test_data(data_path , test_data , starting_index , ending_index):
    count = 0
    test_images = []
    random.shuffle(test_data)
    for index in range(starting_index , ending_index):
        if count < num_images:
            img = cv2.imread(os.path.join(data_path , test_data[index]))
            test_images.append(img)
            count += 1
    return test_images

def pre_process(images):
    preprocessed_images = []
    for image in images:
        resized_image = cv2.resize(image , (224 , 224))
        resized_image = cv2.cvtColor(resized_image , cv2.COLOR_BGR2RGB)
        preprocessed_images.append(resized_image / 255.0)

    return np.array(preprocessed_images).reshape(len(images) , 224 , 224 , 3)

def gen_test_label(is_face , num_images):
    label = [1 , 0] if is_face else [0 , 1]
    labels = [label for i in range(num_images)]
    return labels

def get_predictions(conv_weights_file , bn_weights_file , data_path , test_path , neg_test_path , checkpoint_dir , num_classes , max_images , batch_size , show_image):
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [None , 224 , 224 , 3])
    is_training  = tf.placeholder(dtype = tf.bool)
    model = Darknet19(num_classes , conv_weights_file , bn_weights_file)
    predictions = model.gen_model(tf_input)
    prediction = None
    num_batches = max_images // batch_size
    test_accuracy = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        for batch_id in range(num_batches):
            starting_index = batch_id * batch_size
            ending_index = min(starting_index + batch_size , max_images)
            num_images = ending_index - starting_index
            face_test_images = get_test_data(data_path , test_data , starting_index , ending_index)
            face_test_labels = gen_test_label(True , len(face_test_images))
            neg_test_images = get_test_data("" , neg_test_data , starting_index , ending_index)
            neg_test_labels = gen_test_label(False , len(neg_test_images))

            test_images = face_test_images + neg_test_images
            test_labels = face_test_labels + neg_test_labels
            prediction = sess.run(predictions , feed_dict = {tf_input : pre_process(test_images) , is_training : False})
            test_accuracy += utils.get_accuracy(prediction , test_labels)

            if show_image:
                for index in range(prediction.shape[0]):
                    pred_index = np.argsort(prediction[index])[::-1][0]
                    pred_class = "FACE" if pred_index == 0 else "NON FACE"
                    cv2.imshow(pred_class , cv2.resize(test_images[index] , (416 , 416)))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        test_accuracy /= num_batches

    return test_accuracy

def load_random_images(image_folder):
    images = []
    for image_name in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder , image_name))
        images.append(img)
    return images

def test_random_images(image_folder):
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [None , 224 , 224 , 3])
    is_training  = tf.placeholder(dtype = tf.bool)
    model = Darknet19(num_classes , conv_weights_file , bn_weights_file)
    predictions = model.gen_model(tf_input)
    prediction = None
    test_images = load_random_images(image_folder)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        prediction = sess.run(predictions , feed_dict = {tf_input : pre_process(test_images) , is_training : False})
        for index in range(prediction.shape[0]):
            pred_index = np.argsort(prediction[index])[::-1][0]
            pred_class = "FACE" if pred_index == 0 else "NON FACE"
            cv2.imshow(pred_class , cv2.resize(test_images[index] , (416 , 416)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__" :
    conv_weights_file = "./data/conv_weights.pickle"
    bn_weights_file = "./data/bn_weights.pickle"
    data_path = "./data/celeba/original_images/img_celeba/"
    test_path = "./data/celeba_test_file.pickle"
    neg_test_path = "./data/neg_samples_test.pickle"
    checkpoint_dir = "./saved_models/"
    num_images = 1000
    batch_size = 50
    num_classes = 2
    test_data = load_pickle(test_path)
    neg_test_data = load_pickle(neg_test_path)
    test_accuracy = get_predictions(conv_weights_file , bn_weights_file , data_path , test_path , neg_test_path , checkpoint_dir , num_classes , num_images , batch_size , False)
    print("*************  TEST accuracy : " , (test_accuracy * 100) , "% *****************")

    ############### Testing on random images containing multiple faces #######################
    image_folder = "./multiple_face_images/"
    test_random_images(image_folder)
    print("Try and Try untill you succeed.")
