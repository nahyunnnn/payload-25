import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
# import mpld3
# import streamlit.components.v1 as components
st.set_page_config(page_title='Acceleration vs Pressure')
st.title('Acceleration vs Pressure graph')
data1 = st.file_uploader('CSV file')

if data1 is not None:
    acc_press_data = pd.read_csv(data1)
    time_data = acc_press_data['Time']
    acc_data = np.array(acc_press_data['Acceleration'])
    press_data = np.array(acc_press_data['Pressure'])

# st.title("3D Graph")
# if 'wavelengths' in st.session_state:
#     wave_data = st.session_state['wavelengths']

# if 'intensity' in st.session_state:
#     int_data = st.session_state['intensity']
#     acc_data = np.array(int_data['Acceleration'])

#     cols = np.arange(2,20,step = 1)
#     z = int_data.iloc[:,cols]
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




    # def plot(df, long, lat):

    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    #     X = long
    #     Y = lat
    #     X, Y = np.meshgrid(X, Y)
    #     Z = df

    #     #Plot surface.
    #     surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
    #                         linewidth=0, antialiased=False)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     ax.zaxis.set_major_formatter('{x:.02f}')

    #     # Add a color bar
    #     fig.colorbar(surf, shrink=0.5, aspect=5)

    #     return fig
    st.pyplot(plot(acc_data, time_data,press_data))

    # st.write('Upload a CSV file in the main page to get started!')