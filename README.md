# Face-Classification-Model
A face - Non_face classification model developed from scratch , which used darknet 19 structure.  
This model also serves as a backbone for classifier in [***face detection***](https://github.com/yogeesh-agarwal/Yolov2-Face_Detection) model which uses yolov2 structure for face detection trained and tested on [***wider face dataset***](http://shuoyang1213.me/WIDERFACE/).  
By default training and testing dataset for this classification model is ***Celeb dataset***.

**Requirements**  
1. python 3.6+
2. Tensorflow 1.14+
3. numpy
4. imgaug

***while cloning this repo please make sure git lfs client is installed in local machine or else install it from [here](https://git-lfs.github.com/)***  
*requirements.txt* can also be used to create the exact environment in which this model was created , trained , and tested in.  
*Note : please download the [**celeb dataset**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place it in data folder , as it will be used for training and inference.*

**Training**  
* This model is capable of distinguishing between an image containing face and one without any face in it, Hence it has been trained on dataset containing faces (positive samples) and any other image which doesnt have any faces (nagetive samples).  
Thus for positive samples , by default celeba dataset is used for training , where as , dataset of negative samples is created by running the script which searches the internet for images which random object in it (not faces) and download them as negative samples.  
To train on any other dataset , download the dataset and place it under data folder. Create a per_processing script for it which in-turn creates training and testing pickle files contanining names of the instances. Then we can use these pickle file in the test_preprocessing file to see if dataset is working fine or not.
This can be done  by running below command :  ```python gen_neg_samples.py```  

* After the dataset is ready for training , one can train the model by running the training script which uses the hyperparameter like learning rate , num_batches , batch_size , e.t.c to train the model using the training pickle file created in above step. This script also uses Cyclic Learning Rate as a learning rate scheduler to make the training process faster and convergence easier. The method used in for this scheduling is taken from [this paper](https://arxiv.org/pdf/1506.01186.pdf) , and the kind of scheduling used is referred as ***triangular scheduling*** in the paper which is default.    
As an observation , the loss incurred as well as training accuracy in the starting few iterations of training itself yields very good numbers , and hence training for just very few epochs would just be enough to train the model to classify between face and non face very efficiently.  
Following command can we used to train the model : ``` python train.py```  

* Once training is done , the model is saved in the saved_models directory which contains the model in frozen state and can be used to load the model for either resume the training where we left it or use it for inference.  

* We can also extract the weights and biases from the frozen model , using a script which gives us the weights and batchnorm weights to be used in the fresh model of same structure , also being used in [face detection](https://github.com/yogeesh-agarwal/Yolov2-Face_Detection) model.

**Testing**  
Testing can be done using either of two methods listed below :  
* using frozen model : the model which was saved during training can be used to as it is for testing using load_model moudle of tensorflow using the exact same variables which were stored while training. The testing dataset is again the celeba dataset , and is totally different from training set.   
***The testing accuracy for celeba dataset comes to be around 95%.***  
Command to test using above method : ```python test.py```  

* using extracted weights : One can also test the model by creating fresh model which uses the same structure as darknet19 and uses the extracted weights and batchnorm vars , to infer. This gives us more options in terms of testing dataset and also is important testing phase before using these weights for yolov2 base classifier.  
Current implementation for this method uses above mentioned celeba dataset for accurayc calculation , and also uses bunch of random downloaded images of faces to see the results of random images.
***The testing accuracy comes to be around 96%.***  
Command to test using above method : ```python inference.py```  
