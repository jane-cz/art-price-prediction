import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import json
import torch, torchvision
from torchvision import models, transforms
import os
from datetime import datetime

from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score, recall_score,
                            mean_absolute_error, mean_squared_error, r2_score)
# From https://github.com/vindruid/streamlit-leaderboard
st.set_option('deprecation.showfileUploaderEncoding', False)

# funtions
def relative_time(t_diff):
    days, seconds = t_diff.days, t_diff.seconds
    if days > 0: 
        return f"{days}d"
    else:
        hours = t_diff.seconds // 3600
        minutes = t_diff.seconds // 60
        if hours >0 : #hour
            return f"{hours}h"
        elif minutes >0:
            return f"{minutes}m"
        else:
            return f"{seconds}s"

def get_leaderboard_dataframe(csv_file = 'leaderboard.csv', greater_is_better = True):
    df_leaderboard = pd.read_csv('leaderboard.csv', header = None)
    df_leaderboard.columns = ['Username', 'Score', 'Submission Time']
    df_leaderboard['counter'] = 1
    df_leaderboard = df_leaderboard.groupby('Username').agg({"Score": "max",
                                                            "counter": "count",
                                                            "Submission Time": "max"})
    df_leaderboard = df_leaderboard.sort_values("Score", ascending = not greater_is_better)
    df_leaderboard = df_leaderboard.reset_index()                                                    
    df_leaderboard.columns = ['Username','Score', 'Entries', 'Last']
    df_leaderboard['Last'] = df_leaderboard['Last'].map(lambda x: relative_time(datetime.now() - datetime.strptime(x, "%Y%m%d_%H%M%S")))
    return df_leaderboard

def exe(): 
    # Title
    st.title("Competition Leaderboard")

    # Showing Leaderboard 
    # st.header("Leaderboard")
    if os.stat("leaderboard.csv").st_size == 0:
        st.text("NO SUBMISSION YET")
    else:
        df_leaderboard = get_leaderboard_dataframe(csv_file = 'leaderboard.csv', greater_is_better = True)
        st.dataframe(df_leaderboard)
    
    # Username Input
    username = st.text_input("Username", value = "billy", max_chars= 20,)
    username = username.replace(",","") # for storing csv purpose
    st.subheader(f"Hi {username}! Upload your painting to enter the competition!")
    
    upload = st.file_uploader("Upload Submission JPG File",type='jpg')
    if st.button("SUBMIT"):
        if upload is None:
            st.text("UPLOAD FIRST")
        else:
            # save submission
            # if image is colored (RGB)
            wimage = Image.open(upload)
            st.image(wimage, caption='Your Image.', use_column_width=True)
            image = Image.new("RGB", wimage.size)
            image.paste(wimage)
            uploaded_file = np.asarray(image)
            if(np.ndim(uploaded_file)==3 and uploaded_file.shape[2] == 3):
                
              # reshape it from 3D matrice to 2D matrice
              imageMat_reshape = uploaded_file.reshape(uploaded_file.shape[0], -1)
            # if image is grayscale
            else:
              # remain as it is
              imageMat_reshape = uploaded_file
            
            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_submission = f"submission/sub_{username}__{datetime_now}.csv"
            # saving matrice to .csv file
            np.savetxt(filename_submission, imageMat_reshape)
            
            # retrieving matrice from the .csv file
            # loaded_2D_mat = np.loadtxt(filename_submission)
            # reshaping it to 3D matrice
            # if(np.ndim(uploaded_file)==3):
            #    loaded_mat = loaded_2D_mat.reshape(loaded_2D_mat.shape[0],loaded_2D_mat.shape[1]//uploaded_file.shape[2],uploaded_file.shape[2])
            # loaded_mat = loaded_mat.astype(np.uint8)
            # image = Image.fromarray(loaded_mat, 'RGB')
            # image = st.image(image, caption='Your Image.', use_column_width=True)
            
            # calculate score
            load_clf = torch.load('model/model_4_resnet18.pkl', map_location=torch.device('cpu'))
            xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            input_tensor = xform(image)
            batch_t = input_tensor.unsqueeze(0)
            load_clf.eval()
            out=load_clf(batch_t)
            score = np.exp(out.item())
            score = round(score, 2)
            st.subheader(f"Estimated Price: ${score}")
            st.subheader("Refresh the page to see your submission on the leaderboard!")
            # save score
            with open("leaderboard.csv", "a+") as leaderboard_csv:
                leaderboard_csv.write(f"{username},{score},{datetime_now}\n")