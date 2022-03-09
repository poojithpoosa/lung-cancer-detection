# Lung cancer detection system using transfer learning
 
I have developed an automatic system that helps in the detection of abnormalities in the lung and helps to diagnose cancer using machine learning. 

For the project, the LC25000 dataset is considered, which has a RBG image of three lung cancer diseases. 
Link : https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images


Datagenerator is used for creating a pipeline for the model.

Feature extraction is done by two pre-trained models, which are the VGG19 and ResNet50 models. 

The classification task is done by a simple two-density layer model with 256 and 128 bit sizes and a dropout layer of 0.2 frequency. 
The models gave an excellent accuracy score of 97.97% for the VGG19 model and the ResNet50 model with 99%.
