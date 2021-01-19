import os
import sys
import utils
import numpy as np
import tensorflow as tf
from model import Darknet19
from pre_processing import DataGenerator

def save(session , saver , checkpoint_dir , step):
    dir = os.path.join(checkpoint_dir , "darknet19")
    saver.save(session , dir , global_step = step)
    print("model saved at {} for epoch {}".format(dir , step))

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

def train(input_size ,
          batch_size,
          num_epochs,
          data_path ,
          pos_train_file,
          neg_train_file,
          val_file,
          is_norm,
          is_augment,
          num_classes,
          save_dir,
          logs_dir):

    input = tf.placeholder(dtype = tf.float32 , shape = [None , input_size , input_size , 3])
    ground_truth = tf.placeholder(dtype = tf.float32 , shape = [None , 2])
    is_training = tf.placeholder(dtype = tf.bool , shape = [])
    g_step = tf.get_variable("global_step" , trainable = False , initializer = 0)

    data_generator = DataGenerator(input_size , data_path , pos_train_file , neg_train_file , is_norm , is_augment , val_file = val_file)
    model = Darknet19(num_classes , is_training)
    predictions = model.gen_model(input)
    classification_loss , merged_summary_op = utils.class_loss(predictions , ground_truth)

    l_rate = utils.clr_scheduler(global_step = g_step , current_lr = 0.0001 , max_lr = 0.001 , step_size = 10)
    optimizer = tf.train.AdamOptimizer(learning_rate = l_rate , name = "optimizer")
    train_step = optimizer.minimize(classification_loss , global_step = g_step)

    val_starting_index = 0
    val_ending_index = 100
    val_batch_size = 10
    val_num_batches = (val_ending_index - val_starting_index) // val_batch_size
    num_iter = data_generator.num_train_instances // batch_size
    print("num iter : " , num_iter)
    summary_writer = tf.summary.FileWriter(logs_dir , graph = tf.get_default_graph())
    l_rates = []
    losses = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver(max_to_keep = 4)
        if_checkpoint , epoch_start = load_model(sess , saver , save_dir)
        if if_checkpoint:
            print("Loading partially trained model as epoch : " , epoch_start)
        else:
            print("training from scratch")

        curr_max_loss = 1e+9
        for epoch in range(epoch_start , num_epochs):
            iter_loss = 0
            for iter in range(num_iter):
                current_image_batch , current_label_batch = data_generator.load_train_data(batch_size,  iter)
                current_loss , _ , summary_opr = sess.run([classification_loss , train_step , merged_summary_op] , feed_dict = {input : current_image_batch, ground_truth : current_label_batch , is_training : True})
                c_lr = sess.run(l_rate)
                print(c_lr)
                iter_loss += current_loss
                summary_writer.add_summary(summary_opr , epoch)

            current_gstep = tf.train.global_step(sess , g_step)
            curr_lr = sess.run(l_rate)
            per_epoch_loss = iter_loss / num_iter
            print(f"epoch : {epoch} , loss : {per_epoch_loss}")
            l_rates.append(curr_lr)
            losses.append(per_epoch_loss)
            if epoch % 5 == 0:
                accuracy = 0
                for batch_index in range(val_num_batches):
                    curr_si = val_starting_index + batch_index*val_batch_size
                    curr_ei = min(curr_si + val_batch_size , val_ending_index)
                    current_val_images , current_val_labels = data_generator.load_val_data(curr_si, curr_ei)
                    val_preds = sess.run(predictions , feed_dict = {input : current_val_images , is_training : False})
                    accuracy += utils.get_accuracy(val_preds , current_val_labels)
                avg_accuracy = accuracy / val_num_batches
                print("current_epoch : " , epoch , "\tcurrent_gstep = " , current_gstep , "\tcurrent_lr = " , curr_lr , "\tcurrent_loss : " , per_epoch_loss , " \tAccuracy : " , avg_accuracy)

            if epoch % 250 == 0:
                np.save("./data/lr.npy" , l_rates)
                np.save("./data/losses.npy" , losses)

            if(epoch % 10 == 0 and per_epoch_loss < curr_max_loss):
                curr_max_loss = per_epoch_loss
                save(sess , saver , save_dir , epoch)

            if(per_epoch_loss >= 0.01 and per_epoch_loss <= 0.04):
                save(sess , saver , save_dir , epoch)
                print("Stopping early")
                sys.exit(0)

    print("training Completed")


def main():
    input_size = 224
    batch_size = 16
    num_epochs = 100
    data_path = "./data/celeba/original_images/img_celeba/"
    pos_train_file = "./data/celeba_train_file.pickle"
    neg_train_file = "./data/neg_samples.pickle"
    val_file = "./data/celeba_val_file.pickle"
    is_norm = True
    is_augment = True
    num_classes = 2
    save_dir = "./saved_models"
    logs_dir = "./logs/"

    train(input_size,
          batch_size,
          num_epochs,
          data_path,
          pos_train_file,
          neg_train_file,
          val_file,
          is_norm,
          is_augment,
          num_classes,
          save_dir,
          logs_dir)

if __name__ == "__main__":
    main()
