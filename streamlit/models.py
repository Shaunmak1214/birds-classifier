from re import S
import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

def app():
    st.title("Models Trained")
     
    st.subheader("There are a total of 3 models trained in this project")
    st.write(" ")
    st.table(pd.DataFrame({
        'Model': ['VGG16', 'Mobile Net', 'VIT'],
    }))
    st.write("The model is trained on primarily two metrics, **Learning Rate** of 0.001 and 0.0001. All models are trained on **epoch** of 30 and **batch size** of 32")
    
    tab1, tab2, tab3, tab4 = st.tabs(["VGG16", "Mobile Net", "VIT", "Utils"])
    
    with tab1: 
      st.title("Model 1 - VGG16")
      st.write("VGG16 is a convolutional neural network that is 16 layers deep. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. The network has an input image size of 224-by-224.")
      
      st.subheader("Model Architecture")
      st.image('./diagrams/model_architecture/vgg16_architecture.png')
      
      st.columns(2, gap = 'small')
      st.subheader("Model Performance - Learning Rate 0.001")

      st.image('./diagrams/vgg16/0.001lr_graphs/optimizers_comparison.png')
      st.image('./diagrams/vgg16/0.001lr_graphs/Adam_acc.png')
      st.image('./diagrams/vgg16/0.001lr_graphs/Adam_loss.png')
      st.image('./diagrams/vgg16/0.001lr_graphs/SGD_acc.png')
      st.image('./diagrams/vgg16/0.001lr_graphs/SGD_loss.png')
      
      st.write(" ")
      st.subheader("Model Performance - Learning Rate 0.0001")
      
      st.image('./diagrams/vgg16/0.0001lr_graphs/lr0.0001_optimizer_comparison.png')
      st.image('./diagrams/vgg16/0.0001lr_graphs/lr0.0001_Adam_acc.png')
      st.image('./diagrams/vgg16/0.0001lr_graphs/lr0.0001_Adam_loss.png')
      st.image('./diagrams/vgg16/0.0001lr_graphs/lr0.0001_SGD_acc.png')
      st.image('./diagrams/vgg16/0.0001lr_graphs/lr0.0001_SGD_loss.png')
      
      st.write(" ")
      st.subheader("Model Performance Table - By Optimizers")
      st.table(pd.DataFrame({
          'Optimizer': ['SGD', 'ADAM'],
          "Best accuracy": ["0.96", "0.96"],
          "Best Accuracy Epochs": ["10", "19"],
          "Best loss": ["0.17", "0.17"],
          "Best Loss Epochs": ["27", "19"]
      }))
      
    with tab2: 
      st.title("Model 2 - Mobile Net")
      st.write(" Mobile Net is a convolutional neural network that is 16 layers deep. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. The network has an input image size of 224-by-224.")
      
      st.subheader("Model Architecture")
      st.image('./diagrams/model_architecture/mobile_net_architecture.png')
      
      st.columns(2, gap = 'small')
      st.subheader("Model Performance - Learning Rate 0.001")

      st.image('./diagrams/mobilenet/lr0.001_optimizer_compare.png')
      
      st.write(" ")
      st.subheader("Model Performance - Learning Rate 0.0001")
      
      st.image('./diagrams/mobilenet/lr0.0001_optimizer_comparison.png')
      
      st.write(" ")
      st.subheader("Model Performance Table - By Optimizers")
      st.table(pd.DataFrame({
          'Optimizer': ['SGD', 'ADAM'],
          "Best accuracy": ["0.98", "0.99"],
          "Best Accuracy Epochs": ["16", "10"],
          "Best loss": ["0.11", "0.015"],
          "Best Loss Epochs": ["24", "26"]
      }))
      
    with tab3: 
      st.title("Model 3 - VIT")
      st.write(" VIT is a convolutional neural network that is 16 layers deep. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. The network has an input image size of 224-by-224.")
      
      st.subheader("Model Architecture")
      st.image('./diagrams/model_architecture/vit_architecture.png')
      
      st.write("##### The addition of gaussiun noise layer is to add some noise to the input image. This is to prevent the model from overfitting.")
      
      st.write("*We did not test with two learning rates in this model because the model was showing inconsistent results. We decided to stick with learning rate of 0.05*")
      
      st.image('./diagrams/vit/vit_lr.png')
      
      st.columns(2, gap = 'small')
      st.subheader("Comparing Optimizers")
      st.image('./diagrams/vit/optimizer_comparison.png')
      
      st.write(" ")
      st.subheader("Model Performance Table - By Optimizers")
      st.table(pd.DataFrame({
          'Optimizer': ['SGD', 'ADAM'],
          "Best accuracy": ["0.99", "1.0"],
          "Best Accuracy Epochs": ["0", "1"],
          "Best loss": ["0.05", "0.009"],
          "Best Loss Epochs": ["1", "1"]
      }))

    with tab4: 
      st.write(" ")
      st.write("#### Early Stopping by Loss Vall Accuracy")
      st.write("Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset.")
      
      st.code("""class EarlyStoppingByLossVal(Callback):
  def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
    super(Callback, self).__init__()
    self.monitor = monitor
    self.value = value
    self.verbose = verbose
      

  def on_epoch_end(self, epoch, logs={}):
    current = logs.get(self.monitor)
    if current is None:
        warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

    if current < self.value:
        if self.verbose > 0:
            print("Epoch %05d: early stopping THR" % epoch)
        self.model.stop_training = True """)
      
      st.write(" ")
      st.write("#### Learning Rate Scheduler")
      st.write(" Learning rate scheduling is a method that reduces the learning rate during training by a factor of 2 every 5 epochs. This has the effect of stabilizing the model training and avoiding large weight updates that can result in unstable training behavior.")
      
      st.code("""def lr_scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp() """)
      

    st.markdown('''Made with ❤️ by **Shaun Mak** ''')
    
     