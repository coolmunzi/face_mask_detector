<b> About the project </b>

Wearing mask is the most effective way to control the spread of Corona virus. Face mask detector applications are in high demand especially to monitor public places like public marketplaces, public offices, malls, government buildings etc  This project aims to create an efficient Face Mask detector application based on deep neural network.

<b> Dataset used for training </b>

I have used the data from [Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection/activity) dataset from Kaggle. The dataset contained annotated images with 3 classes:
    
1. with mask
2. without mask
3. mask worn incorrectly

<b> Technolgy stack used </b>

1. [OpenCV](https://opencv.org/) 
2. [TensorFlow 2 Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
3. [EfficientDet-SSD based Face Mask detector](https://tfhub.dev/tensorflow/efficientdet/d1/1)
4. [Python](https://www.python.org/)


<b> Prerequisites </b>

All the dependencies and required libraries are included in the file _requirements.txt_.

<b> Installation </b>
1. Clone the repo using following command
    $ git clone ~~https://github.com/chandrikadeb7/Face-Mask-Detection.git~~
2. Create a virtual environment and change the directory to the cloned repo
3. Change your directory to the cloned repo and create a Python virtual environment named 'test'
4. Install the dependencies using following command
    $pip install -r requirements.txt
5. To perform face mask detection on images:
    $python face_mask_detector.py --type image --input {_Path to direcory containing images_} --output {_Path to directory where you want to store the images with predictions_}
6. To perform face mask detection on webcam:
    $python face_mask_detector.py --type video --input webcam


<b> License </b>
MIT © https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/LICENSE
