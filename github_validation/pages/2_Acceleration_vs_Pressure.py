import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
from PIL import Image

# Adding logo and configuring page layout
logo = 'https://github.com/nahyunnnn/payload-25/blob/main/github_validation/pages/icons/online_ares_logo.jpeg?raw=true'
st.set_page_config(page_title='Acceleration vs Pressure', page_icon=logo)
st.logo(logo, size = 'large')

st.title('Acceleration vs Pressure graph')
data1 = st.file_uploader('CSV file')

if data1 is not None:
    acc_press_data = pd.read_csv(data1)
    time_data = acc_press_data['Time']
    acc_data = np.array(acc_press_data['Acceleration'])
    press_data = np.array(acc_press_data['Pressure'])

    def plot(x,y,z):

        fig = plt.figure()
        ax = fig.add_subplot(projection ='3d')

        xs = time_data
        ys = acc_data
        zs = press_data
        ax.scatter(xs,ys,zs)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (ms-2)')
        ax.set_zlabel('Pressure')
        return fig
    st.pyplot(plot(time_data, acc_data, press_data))
