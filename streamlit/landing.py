from re import S
import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

def app():
    st.title("Birds Classification")
    st.write("In this project we aim to make birds classification using Convolutional Neural Network (CNN) and Transfer Learning. We use the dataset from Kaggle which contains 20 different species of birds. The dataset can be found [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).")

    class_names = os.listdir('./dataset/20test')
    
    st.write("*From the big data context, image is an **unstructured data**, therefore the preprocessing methodologies used might be different than what we're familiar with.*")
    
    st.subheader("Introduction to the project")
    st.write("The 9,000-10,000 types or species of birds are arranged in the appropriate taxonomy of kingdom, order, family, genus, and species. The classification of birds helps ornithologists to study the features and habits of birds.")
    
    st.write("The dataset contains 20 different species of birds, each species contains 100 images. The dataset is split into 2 folders, train and test. The train folder contains 20 folders, each folder contains 80 images of the birds. The test folder contains 20 folders, each folder contains 20 images of the birds.")
    
    st.subheader("Dataset - All of the birds species")
    
    fig, ax = plt.subplots(4, 5, figsize=(20, 20))
    ax = ax.flatten()
    for i in range(20):
        image = './dataset/20test/' + class_names[i] + '/1.jpg'
        img = plt.imread(image)
        ax[i].imshow(img)
        ax[i].set_title(class_names[i])
        
    st.pyplot(fig) 
    st.write(" ")
    st.markdown('''Made with ❤️ by **Shaun Mak** ''')