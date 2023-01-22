from re import S
import pandas as pd
import streamlit as st
import numpy as np

import landing
import eda
import models

def renderFooter ():
  st.write(" ")
  st.markdown('''Made with ❤️ by **Shaun Mak** ''')

st.set_page_config(page_title='Birds Classification', page_icon=None, layout="wide",
                    initial_sidebar_state="auto", menu_items=None)
PAGES = {
    "Overview" : landing,
    "EDA & Method Used" : eda,
    "Models": models
}
st.sidebar.title('Birds Classification')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()