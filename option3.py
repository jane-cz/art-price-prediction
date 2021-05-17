# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:02:35 2021

@author: EVA WANG
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch, torchvision
from torchvision import models, transforms

class ArtistModel(torch.nn.Module):
    def __init__(self,n_artists):
        super(). __init__()
        self.cnn = torchvision.models.resnet34(pretrained=True)
        self.embed = torch.nn.Embedding(n_artists, 16)
        self.cnn.fc = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(self.cnn.fc.in_features,32)
        )
        torch.nn.init.xavier_uniform_(self.cnn.fc[1].weight)
        self.fc = torch.nn.Linear(32+16+2,1)
        torch.nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, image, artist, width, height):
          cnn_out = self.cnn(image)
          embed_out = self.embed(artist)
          img_and_a = torch.cat((cnn_out,embed_out.squeeze(1)),dim=1)
          w_and_h = torch.cat((width,height),dim=1)
          fc_in = torch.cat((img_and_a,w_and_h),dim=1)
          output = self.fc(fc_in)
          return output

def exe():
    st.title('Specific Art Price Prediction')
    colnames = ['number', 'artist']
    data = pd.read_csv('Artist_dict - Artist_dict.csv', names=colnames)
    names = data.artist.tolist()
    options = list(range(len(names)))
    value = st.selectbox("Select option", options, format_func=lambda x: names[x])
    test = st.file_uploader("Please upload a Picture of The Painting!", type="jpg")
    dimension = st.beta_columns(2)
    width = dimension[0].number_input("Width (Inch)", value=0.0, min_value=0.0, max_value=100.0, step=0.01,)
    height = dimension[1].number_input("Height (Inch)", value=0.0, min_value=0.0, max_value=100.0, step=0.01,)
    if (st.button("Estimate this drawing")):
        if(value==0):
            st.error('Please select an artist first')
        else:
            if test is not None:
                try:
                    image = Image.open(test)
                    st.image(image, caption='Your Image.', use_column_width=True)
                except:
                    test = test.astype(np.uint8)
                    image = Image.fromarray(test, 'RGB')
                    image = st.image(image, caption='Your Image.', use_column_width=True)
                    
                value=value-1
                rgbimg = Image.new("RGB", image.size)
                rgbimg.paste(image)
                model=ArtistModel(5197)
                load_clf = torch.load('model/expanded_model_resnet34.pkl', map_location=torch.device('cpu'))
                xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                input_tensor = xform(rgbimg)
                batch_t = input_tensor.unsqueeze(0)
                model.load_state_dict(load_clf)
                model.eval()
                out = model(batch_t,torch.LongTensor([value]),torch.Tensor([[width]]),torch.Tensor([[height]]))
                score=np.exp(out.item())
                st.subheader(f"YOUR Price: ${score}")
