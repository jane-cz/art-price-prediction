# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:24:42 2021
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch, torchvision
from torchvision import models, transforms

def exe():
    st.title('Contemporary Art Price Prediction')
    test = st.file_uploader("Please upload a Picture of Your Painting!", type="jpg")
    #dimension = st.beta_columns(2)
    #width = dimension[0].number_input("Width (Inch)", value=0)
    #height = dimension[1].number_input("Height (Inch)", value=0)
    if (st.button("Estimate this drawing")):
        if test is not None:
            try:
                image = Image.open(test)
                st.image(image, caption='Your Image.', use_column_width=True)
            except:
                test = test.astype(np.uint8)
                image = Image.fromarray(test, 'RGB')
                image = st.image(image, caption='Your Image.', use_column_width=True)
                
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            load_clf = torch.load('model/model_4_resnet18.pkl', map_location=torch.device('cpu'))
            xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            input_tensor = xform(rgbimg)
            batch_t = input_tensor.unsqueeze(0)
            load_clf.eval()
            out = load_clf(batch_t)
            score=np.exp(out.item())
            st.subheader(f"YOUR Price: ${score}")
    
    

