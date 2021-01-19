import os
import sys
import cv2
import utils
import pickle
import random
import numpy as np
import tensorflow as tf
from model import Darknet19

def load_pickle(filepath):
    with open(filepath , "rb") as content:
        return pickle.load(content)

def load_model(session , saver , checkpoint_dir):
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        epoch_no = int(ckpt_name[10 : ])
        print("ckpt_name : " , ckpt_name)
        ckpt_file = os.path.join(checkpoint_dir , ckpt_name)
        print("checkpoint_file : " , ckpt_file)
        saver.restore(session , ckpt_file)
        return (True , epoch_no)
    else:
        return (False , 0)

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

def get_predictions(data_path , test_path , neg_test_path , checkpoint_dir , num_classes , max_images , batch_size , display_img):
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [None , 224 , 224 , 3])
    is_training  = tf.placeholder(dtype = tf.bool)
    model = Darknet19(num_classes , is_training)
    predictions = model.gen_model(tf_input)
    prediction = None
    num_batches = max_images // batch_size
    test_accuracy = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        checkpoint = load_model(sess , saver , checkpoint_dir)
        if not checkpoint:
            print("No trained model present")

        for batch_id in range(num_batches):
            starting_index = batch_id * batch_size
            ending_index = min(starting_index + batch_size , max_images)
            num_images = ending_index - starting_index
            face_test_images = get_test_data(data_path , test_data , starting_index , ending_index)
            neg_test_images = get_test_data(neg_test_path , neg_test_data , starting_index , ending_index)
            face_test_labels = gen_test_label(True , len(face_test_images))
            neg_test_labels = gen_test_label(False , len(neg_test_images))

            test_images = face_test_images + neg_test_images
            test_labels = face_test_labels + neg_test_labels
            prediction = sess.run(predictions , feed_dict = {tf_input : pre_process(test_images) , is_training : False})
            test_accuracy += utils.get_accuracy(prediction , test_labels)

            if display_img:
                for index in range(prediction.shape[0]):
                    pred_index = np.argsort(prediction[index])[::-1][0]
                    pred_class = "FACE" if pred_index == 0 else "NON FACE"
                    print(pred_class)
                    cv2.imshow(pred_class , cv2.resize(images[index] , (416 , 416)))
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

        test_accuracy /= num_batches

    return test_accuracy

if __name__ == "__main__" :
    data_path = "./data/celeba/original_images/img_celeba/"
    test_path = "./data/celeba_test_file.pickle"
    neg_test_path = "./data/neg_samples_test.pickle"
    checkpoint_dir = "./saved_models/"
    num_images = 1000
    batch_size = 100
    num_batches = num_images // batch_size
    num_classes = 2
    display_img = False
    test_data = load_pickle(test_path)
    neg_test_data = load_pickle(neg_test_path)
    test_accuracy = get_predictions(data_path , test_path , neg_test_path , checkpoint_dir , num_classes , num_images , batch_size , display_img)
    print("*************  TEST accuracy : " , (test_accuracy * 100) , "% *****************")
    print("Try and Try untill you succeed.")
