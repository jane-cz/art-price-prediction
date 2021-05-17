import option1
import option2
import option3
import leaderboard
import streamlit as st

PAGES = {
    "Price Prediction": option1,
    "Predict with more info": option3,
    "Draw Right Now": option2,
    "Competition Leaderboard": leaderboard
}
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.outdoorpainter.com/wp-content/uploads/2019/09/plein-air-oil-painting-Debra-Latham-Hill-Country-Sunset.jpg")
    }
   .sidebar-container {
        background: url("https://www.outdoorpainter.com/wp-content/uploads/2019/09/plein-air-oil-painting-Debra-Latham-Hill-Country-Sunset.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Select Options", list(PAGES.keys()))
page = PAGES[selection]
page.exe()