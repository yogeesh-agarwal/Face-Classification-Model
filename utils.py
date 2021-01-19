import sys
import numpy as np
import tensorflow as tf

def get_accuracy(prediction , ground_truths):
    tp = 0
    fp = 0
    for pred , gt in zip(prediction , ground_truths):
        index = np.argsort(pred)[::-1][0]
        if (gt[0] and index == 0) or (gt[1] and index == 1):
            tp += 1
        else:
            fp += 1
    accuracy = (tp / (tp + fp))
    return accuracy

def class_loss(predictions , target):
    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = target, logits = predictions))
    print_op = tf.print("class_loss : " , loss , output_stream = sys.stdout)
    with tf.control_dependencies([print_op]):
        tf.summary.scalar("class_loss" , loss)
        merged_summary_op = tf.summary.merge_all()

    return loss , merged_summary_op

def clr_scheduler(global_step , current_lr , max_lr , step_size):
    per_cycle = tf.math.floor(1 + global_step / (2 * step_size))
    x = tf.cast(tf.math.abs((global_step / step_size) - (2 * per_cycle) + 1) , tf.float32)
    learning_rate_delta = current_lr + (max_lr - current_lr) * tf.math.maximum(0. , 1 - x)
    new_learning_rate = current_lr + learning_rate_delta
    return new_learning_rate
