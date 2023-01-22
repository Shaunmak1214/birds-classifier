from re import S
import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

def app():
    st.title("EDA - Exploratory Data Analysis")
    st.write("Explanation of the dataset and the data visualization")
    
    st.markdown("""---""")

    st.subheader("Descriptive Questions") 
    st.write("#### 1. What is the distribution of the dataset?")
        
    st.image('./diagrams/data_description.png')
    
    st.write("##### Inbalance Dataset")
    st.write("As you can visualize on the figure below, the dataset is slighty inbalance, this might  affect the accuracy of the model. We will try to solve this problem by using **data augmentation**.")
    
    st.markdown("""---""")
    st.subheader("Explanatory Questions")  
    st.write("#### 1. What is the brightness distribution of the dataset?")
    
    st.write(" ")
    
    st.subheader("Brightness Distribution")
    st.write("The brightness distribution of the dataset is quite similar, this might affect the accuracy of the model because the model might not be able to distinguish the birds based on the brightness of the image. We will try to solve this problem by using **data augmentation**.")
    
    st.image('./color_distri.png', width=600)
    
    st.write(" ")
    
    st.write("#### 2. What is the color distribution of the dataset?")
    st.write(" ")
    st.subheader("Color Distribution")
    st.write("This is to distinguish whether color is a telling factor in the classification of the birds. In the below exmaple we plotted 5 images of American redstart")
    st.image('./brightness_distri.png', width=1200)
    
    st.write("In this case, we see that for american redstart, the blue channel tend to spike up, this might be a telling factor in the classification of the birds.")
    
    
    st.write("#### But with a scatter plot, this quickly proves us wrong")
    st.image('./cream-cloured-corr.png', width=1200, caption="Cream Coloured WOODPECKER")
    st.image('./kookaburra-corr.png', width=1200, caption="KOOKABURRA")
    st.image('./campo-flicker-corr.png', width=1200, caption="Campo Flicker")
    st.image('./rufuos-motmot-corr.png', width=1200, caption="Rufous Motmot")
    st.image('./white-tailed-tropic-corr.png', width=1200, caption="White-tailed Tropicbird")
        

    st.write(" ")
    st.write("#### 3. What is the size of the images?")
    st.write(" ")
    st.image("./size_demo.png")
    st.subheader("Alot of details")
    
    st.write("What we can do is apply a blur on the image, this will reduce the details of the image and hopefully the model will be able to generalize the image better.")
  
    st.image("./gaussian_blur.png")
    
    st.write("#### ")
    st.markdown("""---""")
    
    st.subheader("Predictive ")  
    st.write("#### 1. What can we predict?")
    st.write("We can predict the species of the birds based on the image.")
    
    st.markdown("""---""")

    st.write("## What can we do to improve the accuracy of the model?")
    
    st.subheader("Brightness Enhancement") 
    st.write("We will try to enhance the brightness of the image by using **Image Data Generator** from tensorflow, by specifying the brightness range, the kernel will randomly adjust the brightness of the image.")
    
    st.subheader("Rotation") 
    st.write("If the dataset provided has no rotation context, the model might not be able to generalize the image. That is why in the preprocessing stage it is important for us to randomize the rotation of the image.")
    
    st.subheader("Zoom Ranges") 
    st.write("CNN don't inherently take care of zoom in images, if during training, enough size variation is uesed, the network will learn to handle it to some extent. However, if the size variation is not enough, the network will not be able to generalize the image. That is why in the preprocessing stage it is important for us to randomize the zoom range of the image.")
    
    st.title("Result of the Data Augmentation") 
    st.image('./brightness_enhancement.png', width=1200)