import cv2
import numpy as np
import pre_processing as pre_process

def test_preprocess():
    batch_size = 10
    num_batch = 2
    batch_generator = pre_process.DataGenerator(input_size = 416,
                                                data_path = "./data/celeba/original_images/img_celeba/",
                                                pos_train_file = "./data/celeba_train_file.pickle",
                                                neg_train_file = "./data/neg_samples.pickle",
                                                is_norm = False,
                                                is_augment = True,
                                                val_file = "./data/celeba_val_file.pickle"
                                                )

    print(batch_generator.num_train_instances , batch_generator.num_val_instances)
    for index in range(num_batch):
        batch_images , batch_labels = batch_generator.load_train_data(batch_size , index)
        # batch_images , batch_labels = batch_generator.load_val_data(0 , 10)
        for i in range(len(batch_images)):
            image = batch_images[i]
            label = batch_labels[i]
            if label[0] == 1:
                print("FACE")
            else:
                print("NON FACE")
            cv2.imshow("image" , image[:,:,::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

test_preprocess()
