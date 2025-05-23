import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os

# Adding logo and configuring page layout
logo = 'https://github.com/nahyunnnn/payload-25/blob/main/github_validation/pages/icons/online_ares_logo.jpeg?raw=true'
st.set_page_config(page_title='Acceleration vs Pressure', page_icon=logo)
st.logo(logo, size = 'large')

st.title('Graphs')

tab1, tab3, tab2 = st.tabs(["Acceleration vs Time", 'Pressure vs Time', "Acceleration vs Pressure" ])

with tab1:
    st.file_uploader('CSV file', key='Graph2')

    if 'Graph2' in st.session_state and st.session_state['Graph2'] is not None:
        data2 = st.session_state['Graph2']
        if data2 is not None:
            acc_press_time_data = pd.read_csv(data2)
            time_data = np.array(acc_press_time_data['Time'])
            def time_index_to_value(dataframe, index):
            # Input : Dataframe containing a time column
            #         Index of time 
            # Output : Value of time in list corresponding to index
                time_column = np.array(dataframe['Time'])
                return time_column[index]
            
            
            def last_time_index(dataframe):
                return pd.Series.last_valid_index(dataframe)
            
            def start_time(dataframe):
                return time_index_to_value(dataframe, 0)

            def end_time(dataframe):
                return time_index_to_value(dataframe,last_time_index(dataframe))

            def time_step(dataframe):
                return time_index_to_value(dataframe, int(start_time(dataframe) + 1)) - start_time(dataframe)

            def time_value_to_index(datalist, time):
                return int(time / time_step(datalist))
            
            min_range = time_data[0]
            max_range = time_data[-1]
            step = time_step(acc_press_time_data)

            min, max = st.slider(label = 'Time range', min_value = min_range, max_value = max_range, value = (min_range, max_range), step = step)

            min_index = time_value_to_index(acc_press_time_data, min)
            max_index = time_value_to_index(acc_press_time_data, max)
            
            press_data = np.array(acc_press_time_data['Force'][min_index:max_index])
            acc_data = np.array(acc_press_time_data['Acceleration'][min_index:max_index])
            
            acceleration = pd.DataFrame({
                'Time': time_data[min_index:max_index],
                'Acceleration': acc_data
            })

            pressure = pd.DataFrame({
                'Time': time_data[min_index:max_index],
                'Pressure': press_data
            })

            fig = px.line(acceleration, x = 'Time', y = 'Acceleration', labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
            # fig.add_scatter(x=pressure['Time'], y = pressure['Pressure'], mode = 'lines', name = 'Pressure')
            # fig = px.line('Acceleration', x = time_data[min_index:max_index], y = acc_data, labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
            # fig = px.line('Pressure', x = time_data[min_index:max_index], y = press_data, labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
            st.plotly_chart(fig)

    else:
        st.write('Upload CSV to get started!')

with tab3:
    st.file_uploader('CSV file', key='Graph3')

    if 'Graph3' in st.session_state and st.session_state['Graph3'] is not None:
        data3 = st.session_state['Graph3']
        if data3 is not None:
            acc_press_time_data = pd.read_csv(data3)
            time_data = np.array(acc_press_time_data['Time'])
            def time_index_to_value(dataframe, index):
            # Input : Dataframe containing a time column
            #         Index of time 
            # Output : Value of time in list corresponding to index
                time_column = np.array(dataframe['Time'])
                return time_column[index]
            
            
            def last_time_index(dataframe):
                return pd.Series.last_valid_index(dataframe)
            
            def start_time(dataframe):
                return time_index_to_value(dataframe, 0)

            def end_time(dataframe):
                return time_index_to_value(dataframe,last_time_index(dataframe))

            def time_step(dataframe):
                return time_index_to_value(dataframe, int(start_time(dataframe) + 1)) - start_time(dataframe)

            def time_value_to_index(datalist, time):
                return int(time / time_step(datalist))
            
            min_range = time_data[0]
            max_range = time_data[-1]
            step = time_step(acc_press_time_data)

            min, max = st.slider(label = 'Time range', min_value = min_range, max_value = max_range, value = (min_range, max_range), step = step)

            min_index = time_value_to_index(acc_press_time_data, min)
            max_index = time_value_to_index(acc_press_time_data, max)
            
            press_data = np.array(acc_press_time_data['Force'][min_index:max_index])
            
            pressure = pd.DataFrame({
                'Time': time_data[min_index:max_index],
                'Pressure': press_data
            })

            fig = px.line(pressure, x = 'Time', y = 'Pressure', labels = {'x':'Time (s)', 'y':'Pressure (Pa)'})
            # fig.add_scatter(x=pressure['Time'], y = pressure['Pressure'], mode = 'lines', name = 'Pressure')
            # fig = px.line('Acceleration', x = time_data[min_index:max_index], y = acc_data, labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
            # fig = px.line('Pressure', x = time_data[min_index:max_index], y = press_data, labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
            st.plotly_chart(fig)

    else:
        st.write('Upload CSV to get started!')

with tab2:
    st.file_uploader('CSV file', key='Acc_vs_Press')
    if 'Acc_vs_Press' in st.session_state and st.session_state['Acc_vs_Press'] is not None:
        data1 = st.session_state['Acc_vs_Press']

        if data1 is not None:
            acc_press_data = pd.read_csv(data1)
            time_data = acc_press_data['Time']
            acc_data = np.array(acc_press_data['Acceleration'])
            press_data = np.array(acc_press_data['Force'])

            def plot(x,y,z):

                fig = go.Figure(data = [go.Scatter3d(
                    x = time_data,
                    y = acc_data,
                    z = press_data,
                    mode = 'markers',
                    marker = dict(
                        size = 5,
                        color = z,
                        colorscale = 'Viridis',

                    )
                )])

                fig.update_layout(
                    scene = dict(
                        xaxis_title = 'Time (s)',
                        yaxis_title = 'Acceleration (ms-2)',
                        zaxis_title = 'Pressure (Pa)'
                    ),
                    margin = dict(l=0, r=0, b=0, t=0)
                )
                return fig 
            st.plotly_chart(plot(time_data, acc_data, press_data))

    else:
        st.write ('Upload CSV to get started!')
