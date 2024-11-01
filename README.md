Fashion is more than just clothes it’s a reflection of who we are and the world around us. From everyday wear to special occasions, 
fashion is everywhere in our lives. With the growing demand for trendy items, the fashion industry is booming. 
But catering to everyone’s tastes is tricky. That’s where fashion recommender systems come in — they’re smart 
tools designed to make shopping easier and more enjoyable for you.


**Process We Can Follow**:

Data Collection: Gather a diverse dataset of fashion items, encompassing various colors, patterns, styles, and categories. 
                 Ensure uniformity in image formats (e.g., JPEG, PNG) and resolutions.
                 
Preprocessing: Develop a preprocessing function to standardize and enhance images for feature extraction. This may involve resizing, 
               normalization, and noise reduction to optimize the quality of input images.
               
Feature Extraction: Choose a pre-trained Convolutional Neural Network (CNN) model such as VGG16, ResNet, or InceptionV3. Utilize transfer 
                    learning to leverage the pre-trained model’s knowledge on large datasets like ImageNet. Extract high-level features 
                    from fashion images using the chosen model.
                    
Similarity Measurement: Define a metric for quantifying the similarity between feature vectors extracted from images. Common methods include 
                        cosine similarity or Euclidean distance. This step is crucial for identifying relevant fashion items based on their 
                        feature similarities.
                        
Ranking and Recommendation: Rank the dataset images based on their similarity to the input image’s features. Recommend the top N items 
                            that exhibit the highest similarity scores. This personalized ranking ensures tailored recommendations aligned 
                            with the user’s preferences.
                            
System Implementation: Develop a comprehensive function that encompasses the entire recommendation process. This includes image preprocessing, 
                       feature extraction, similarity computation, and recommendation generation. Ensure seamless integration of these components 
                       to deliver a user-friendly experience.
                       
By following these steps, we can build a robust fashion recommendation system that harnesses the power of image features extracted using state-of-the-art CNN models.



**Data Collection:**
To build our fashion recommendation system, the first crucial step is gathering a diverse dataset of fashion products. This dataset serves as the foundation for training our model to recognize and recommend stylish items. Fortunately, there are several repositories where such data can be sourced, including Kaggle and Google Datasets. For our project, we’ve selected datasets from Kaggle specifically curated for fashion products:

Fashion Product Images (Small)
Fashion Images
These datasets contain a vast array of fashion items, ranging from clothing and accessories to footwear. By accessing these datasets, we ensure a rich and varied collection that aligns with our recommendation system’s objectives. The techniques employed for data collection remain consistent regardless of the specific recommendation task. Whether it’s suggesting clothing combinations or matching accessories, the fundamental approach remains the same


Preprocessing:
We will utilize ResNet-50 transfer learning techniques for feature extraction in our recommendation system

The preprocessing technique for ResNet-50 involves:

Reading and Resizing: The image is loaded and resized to a standard size, often 224x224 pixels.(User Choice)
Converting and Expanding: The resized image is converted into an array format and expanded to meet ResNet-50’s input requirements.
Preprocessing Input: Additional steps like normalization are applied to align the input data with the model’s expectations
import tensorflow
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model.summary()
The above code prepares our environment for using ResNet-50. It imports TensorFlow and Keras, loads ResNet-50 architecture, and sets up preprocessing functions. We initialize the model with pre-trained weights from ImageNet, excluding fully connected layers, and specify input image shape as (224,224,3) for height, width, and RGB channels. To maintain pre-trained weights, we freeze the model’s layers. The model summary displays its structure, including layer names and output shapes.


how the image passed to ResNet 50 and converted into Feature Vector
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()
We’re improving our model by adding a GlobalMaxPooling2D layer to the existing ResNet-50. This layer extracts the most important features from ResNet-50’s output

After defining the model architecture, the next step is to preprocess the data before feeding it into the model for training and inference.

import cv2
import numpy as np
from numpy.linalg import norm

def extract_feature(img_path,model):
    img = cv2.imread(img_path)
    img=cv2.resize(img,(224,224))
    img =np.array(img)
    expand_img=np.expand_dims(img,axis=0)
    pre_image=preprocess_input(expand_img)
    result=model.predict(pre_image).flatten()
    normalized=result/norm(result)
    return normalized
extract_feature('1550.jpg',model)
#output=
1/1 [==============================] - 0s 288ms/step
array([0.0109979 , 0.00929778, 0.00908274, ..., 0.01481535, 0.00976774,
       0.00913304], dtype=float32)
The extract_feature() function preprocessed the image and then pass the preprocessed image through the model for inference using the predict function. This generates a feature vector representing the image’s characteristics

import os
from tqdm import tqdm
path=r'image folder path'
filename=[]
feature_list=[]

for file in os.listdir(path):
    filename.append(os.path.join(path,file))

for file in filename:
    feature_list.append(extract_feature(file,model))
This code iterates through images in a specified folder, extracts features using the defined model, and stores them in a list. The tqdm library provides a progress bar for tracking the iteration progress.

import pickle
pickle.dump(feature_list,open('featurevector.pkl','wb'))
pickle.dump(filename,open('filenames.pkl','wb'))
The code serializes and saves the extracted feature vectors and corresponding filenames into pickle files

Similarity Measurement:

In our recommendation system, we utilize cosine similarity to measure the similarity between image feature vectors. Cosine similarity quantifies the similarity between two vectors by calculating the cosine of the angle between them. A value of 1 indicates identical vectors, while 0 implies orthogonal vectors. This method enables us to identify images with similar features, facilitating accurate recommendations. For a detailed explanation, you can refer to this article: Cosine Similarity Explained
