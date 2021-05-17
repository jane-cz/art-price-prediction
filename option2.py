# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:28:20 2021

"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch, torchvision
from torchvision import models, transforms
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
import base64

def exe():
    st.title('Contemporary Art Price Prediction')
    st.write("""
     This app estimates how much is your drawing!
    """)
    
    # From https://pypi.org/project/streamlit-drawable-canvas/
    # From https://www.codegrepper.com/code-examples/python/streamlit+download+image
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    #canvas_height = st.sidebar.slider("Canvas height: ", 100, 500, 150)
    #canvas_width = st.sidebar.slider("Canvas width: ", 100, 500, 200)
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="" if bg_image else bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    test = canvas_result.image_data
    #dimension = st.beta_columns(2)
    #width = dimension[0].number_input("Width (Inch)", value=0)
    #height = dimension[1].number_input("Height (Inch)", value=0)
    
    # Do something interesting with the image data and paths
    #if canvas_result.image_data is not None:
    #    st.image(canvas_result.image_data)
    #if canvas_result.json_data is not None:
    #    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    
    if (st.button("Estimate this drawing")):
        if test is not None:
            test = test.astype(np.uint8)
            image = Image.fromarray(test, 'RGBA')
            image = image.convert('RGB')
            st.image(image, caption='Your Image.', use_column_width=True)
                
            result = Image.fromarray(test,mode="RGBA")

            def get_image_download_link(img):
                """Generates a link allowing the PIL image to be downloaded
                in:  PIL image
                out: href string
                """
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                href = f'<a href="data:file/png;base64,{img_str}">Download result</a>'
                return href
                    
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        
            load_clf = torch.load('model/model_4_resnet18.pkl', map_location=torch.device('cpu'))
            xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            input_tensor = xform(image)
            batch_t = input_tensor.unsqueeze(0)
            load_clf.eval()
            out = load_clf(batch_t)
            score=np.exp(out.item())
            st.subheader(f"YOUR Price: ${score}")
