# **Brain Tumor Classification using Deep Learning & Transfer Learning**

This project develops a deep learning model to classify brain MRI scans into four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor. The final model utilizes the **ResNet50 architecture** and **Transfer Learning** to achieve **91.7% accuracy** on the test dataset.

This repository documents the end-to-end process, from establishing a baseline CNN to implementing a high-performance transfer learning solution and diagnosing model generalization issues.

## **Table of Contents**

* [Project Overview](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#project-overview) 
* [Repository Structure](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#repository-structure)  
* [Dataset](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#dataset)  
* [Methodology](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#methodology)  
* [Results](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#results)  
* [Usage](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#usage)  
* [Known Issues & Future Work](https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet/blob/main/Readme.md#known-issues--future-work)

## **Project Overview**

The goal of this project is to build and evaluate a robust classifier for identifying brain tumors from MRI scans. The project demonstrates a realistic machine learning workflow:

1. **Baseline Model:** A custom Convolutional Neural Network (CNN) was first built to establish a performance baseline ([Notebook 1](https://www.google.com/search?q=./notebooks/1_baseline_cnn_model.ipynb)).  
2. **Performance Improvement:** To significantly boost accuracy, a **Transfer Learning** approach was implemented using the pre-trained **ResNet50** model ([Notebook 2](https://www.google.com/search?q=./notebooks/2_transfer_learning_resnet50.ipynb)).  
3. **Inference Script:** A prediction script was developed to classify single, external images using the trained ResNet50 model ([Notebook 3](https://www.google.com/search?q=./notebooks/3_prediction_with_resnet50.ipynb)).

## **Repository Structure**

brain-tumor-classification/  
│  
├── notebooks/                \# Contains all Jupyter notebooks for the project.  
│   ├── 1\_baseline\_cnn\_model.ipynb  
│   ├── 2\_transfer\_learning\_resnet50.ipynb  
│   └── 3\_prediction\_with\_resnet50.ipynb  
│  
├── saved\_model/              \# Contains the trained model file.  
│   └── resnet50\_tumor\_classifier.h5
|   └── brain\_tumor\_classifier.h5
│  
├── images/                   \# Contains result plots and banner images.  
│   ├── confusion\_matrix.png  
│   ├── training\_plots.png  
│   └── project\_banner.png  
│   
└── README.md                 \# This README file.  


## **Dataset**

The project utilizes the "Brain Tumor MRI Dataset" available on Kaggle. It contains over 7,000 MRI images across four classes.

* **Dataset Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## **Methodology**

The final model uses a **ResNet50** base, pre-trained on ImageNet. The convolutional layers were frozen to act as a powerful feature extractor, and a new classification head was added and trained on the brain tumor dataset. This approach leverages existing knowledge to achieve high accuracy.

## **Results**

The implementation of Transfer Learning resulted in a significant performance increase. The final ResNet50-based model achieved a **validation accuracy of 91.69%**, a notable improvement over the baseline CNN's 78.34%.

### **Model Performance Comparison**

#### **Baseline CNN Model (78.34% Accuracy)**

| Class | Precision | Recall | F1-Score |
| :---- | :---- | :---- | :---- |
| glioma | 0.94 | 0.65 | 0.77 |
| meningioma | 0.80 | 0.42 | 0.55 |
| notumor | 0.84 | 1.00 | 0.91 |
| pituitary | 0.65 | 1.00 | 0.79 |

#### **Final ResNet50 Model (91.69% Accuracy)**

| Class | Precision | Recall | F1-Score |
| :---- | :---- | :---- | :---- |
| glioma | 0.93 | 0.86 | 0.89 |
| meningioma | 0.83 | 0.84 | 0.83 |
| notumor | 0.95 | 0.98 | 0.96 |
| pituitary | 0.95 | 0.98 | 0.96 |

The ResNet50 model shows much more balanced performance, especially improving the recall for **glioma** (65% to 86%) and **meningioma** (42% to 84%) tumors, which were weak points for the baseline model.

#### **Training History (ResNet50)**

<img width="1388" height="490" alt="history_plots" src="https://github.com/user-attachments/assets/910f4cf1-af51-449d-85af-ad01c93bd283" />

These plots show the model's accuracy and loss on the training and validation sets over each epoch. The charts indicate a good fit, with the model learning effectively without significant overfitting.


#### **Confusion Matrix (ResNet50)**

<img width="788" height="701" alt="resnet_confusion_matrix" src="https://github.com/user-attachments/assets/30d137a8-447f-408a-b0f8-d257a853a937" />

The confusion matrix provides a detailed breakdown of the model's predictions, showing that it performs well across all four classes.

## **Usage**

1. Clone this repository:  
   git clone \(https://github.com/PranitaAnnaldas/brain-tumor-classification-cnn-resnet)

2. To train the models, run the notebooks in the notebooks/ directory in sequential order.  
3. To make a prediction, use the 3\_prediction\_with\_resnet50.ipynb notebook. Make sure the resnet50\_tumor\_classifier.h5 file is in the saved\_model/ directory.

## **Known Issues & Future Work**

The current model, while highly accurate on the test set, currently underperforms on external images from the internet. A diagnostic analysis identified a **data preprocessing mismatch** between the training generator and the manual inference script.

* **Next Step:** The immediate plan is to retrain the ResNet50 model using a simpler, more robust preprocessing method (e.g., rescale=1./255) to ensure a consistent pipeline and improve its real-world generalization.
