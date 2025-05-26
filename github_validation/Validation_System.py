import streamlit as st
import altair as alt
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tempfile
import plotly.graph_objects as go
from PIL import Image
import plotly.express as px
import math

# Adding logo and configuring page layout
logo = 'https://github.com/nahyunnnn/payload-25/blob/main/github_validation/pages/icons/online_ares_logo.jpeg?raw=true'
st.set_page_config(layout='wide', page_title='Main', page_icon=logo)
st.logo(logo, size = 'large')

alt.themes.enable("dark")

st.markdown("""
    <!-- Load Source Sans Pro font -->
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap" rel="stylesheet">

    <style>
        /* Apply Source Sans Pro globally */
        html, body, [data-testid="stAppViewContainer"], [data-testid="block-container"] {
            font-family: 'Source Sans Pro', sans-serif;
        }

        /* Adjust main page container padding */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    line_image = st.file_uploader('Upload your line image')
    if line_image:
        st.session_state['line_image'] = line_image
    csv_data = st.file_uploader('Upload your data CSV', type=['csv'])
    if csv_data:
        st.session_state['csv_data'] = csv_data


if 'csv_data' in st.session_state:
    csv_variable = st.session_state['csv_data']

    csv_variable.seek(0)
    df = pd.read_csv(csv_variable)

    pressure_list = df['Force']
    acc_list = df['Acceleration']
    time_list = df['Time']
    acc_press_time_data = df

############################################
#       Container formatting               #
#       col1       col2          col3      #
#      _______________________________     #
#      |     |     |     |     |     |     #
#      |_____|_____|_____|_____|     |     #
#      |     |                 |     |     #
#      |_____|                 |     |     #
#      |     |                 |     |     #
#      |_____|_________________|_____|     #
##########################################

col1, col2, col3 = st.columns([1,3,1])

with col2:
    con2_1 = st.container()
    con2_2 = st.container()

with con2_1:
    col2_1_1, col2_1_2, col2_1_3 = st.columns([1,2,0.1])
    with col2_1_1:
        st.write('Original line image')
    with col2_1_2:
        st.write('Line image transform')

with con2_2:
    col2_2_1, col2_2_2 = st.columns([3, 0.1])

with col3:
    st.container(height = 20, border = False)

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ["#338F59", "#22412F"]
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [1-input_response, input_response]
        })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [1, 0]
        })
    
    plot = alt.Chart(source).mark_arc(innerRadius=57, cornerRadius=2).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            #domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
                        ).properties(width=150, height=150)
    
    text = alt.Chart(source).mark_text(
    align='center',
    color="#FFFFFF",
    font="Source Sans Pro",
    fontSize=20,
    fontWeight=500
        ).encode(
            text=alt.value(f'{int(100 * input_response)}%')
        ).properties(width=150, height=150)
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=50, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=150, height=150)
    return plot_bg + plot + text

# Validation system code   
if 'line_image' in st.session_state:

    if line_image:
        with tempfile.NamedTemporaryFile(delete=False,suffix = '.jpg') as temp_image:
            temp_image.write(line_image.getbuffer())
            temp_image.seek(0)
            col2_1_1.image(temp_image.name)
        img = cv.imread(temp_image.name, cv.IMREAD_GRAYSCALE)
        if img is None:
            st.write('File read unsuccesful. Please check file type.')
    
        imgBlur = cv.GaussianBlur(img, (15,15), 3)
        thresholded_image = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)        
        cannyEdge =cv.Canny(imgBlur, 50, 180)
        
        def calculate_straightness(edge_image, tolerance):
            edge_points = np.column_stack(np.where(edge_image > 0))
            if len(edge_points) < 2:
                return 0, 0, 0
            
            #Least squares regression
            x = edge_points[:,1]
            y = edge_points[:,0]

            A = np.vstack([x, np.ones_like(x)]).T
            m,b = np.linalg.lstsq(A,y,rcond=None)[0]

            distances = np.abs(m*x+b - y)/np.sqrt(m**2+1)
            normal_distances =np.clip(distances/tolerance,0,1)
            
            # draw line of best fit:
            
            # con2_1.latex(r"y")
            x_min, x_max = int(np.min(x)), int(np.max(x))
            y_min, y_max = int(m*x_min + b), int(m*x_max+b)
            cv.line(img, (x_min, y_min), (x_max, y_max), (0,0,255), 2)

            score = 1-np.mean(normal_distances)
            return score, m, b

        def houghLineTransform():

            fig = plt.figure(figsize=(18,10))
            plt.subplot(121)
            plt.imshow(img)            
            plt.subplot(122)

            plt.imshow(cannyEdge)
        
            # Accuracy score calculations 
            m = calculate_straightness(cannyEdge, tolerance = 100)[1]
            b = calculate_straightness(cannyEdge, tolerance = 100)[2]

            accuracyscore = 1 - math.atan(abs(m)) / math.pi 

            col3.markdown(f"""
                <div data-testid="stMetric" style="
                    background-color: #262730;
                    text-align: center;
                    padding: 15px 10px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0);
                    margin-bottom: 1rem;
                    font-family: 'Source Sans Pro', sans-serif;
                    color: white;
                ">
                    <div data-testid="stMetricLabel" style="color: #FFFFFF; font-size: 0.9rem; margin-bottom: 0.25rem;">
                        Line of best fit
                    </div>
                    <div style="
                        font-size: 1rem; 
                        font-weight: 300; 
                        color: white; 
                        background-color: #262730;
                        padding: 0; 
                        margin: 0;
                        border-radius: 0;
                    ">
                        <em>y = {m:.2f}x + {b:.2f}</em>
                    </div>
                </div>
            """, unsafe_allow_html=True)
   

            distResol = 1            
            angleResol = np.pi/180
            threshold = 250
            lines = cv.HoughLines(cannyEdge,distResol, angleResol, threshold)
            k=500
            if lines is not None:                
                for curline in lines:
                    rho,theta = curline[0]
                    dhat = np.array([[np.cos(theta)],[np.sin(theta)]])
                    d = rho*dhat
                    lhat = np.array([[-np.sin(theta)],[np.cos(theta)]])
                    p1 = d + k*lhat
                    p2 = d - k*lhat
                    p1 = p1.astype(int)
                    p2 = p2.astype(int)
                    cv.line(img, (p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (0,0,0), 10)
                    # Line drawn 
                    ''' xcos(theta) + ysin(theta) = p                       
                        y = -cos(theta)/sin(theta)x + p/sin(theta) 
                    '''
                    if np.sin(theta) != 0:
                        m = -np.cos(theta)/np.sin(theta)
                        b = rho/np.cos(theta)
                        # con_2.write(f"Line Equation: y = {m:.2f}x+{b:.2}")
                col2_1_1.write("Hough Lines detected,") #Score automatically 1 
            else:
                score = calculate_straightness(cannyEdge, tolerance = 100)[0]
                if score is not None:
                    if float(score) > 0.5:
                        global indicator_color 
                        indicator_color ='lightgreen'
                    else:
                        indicator_color = 'red'
                    colors = [indicator_color,'white']
                    values3_1 = [score, 1-score]
                    with col1:
                        st.write('Score')
                        # figs3_1 = go.Figure(data=[go.Pie(values=values3_1, hole=0.6)])
                        # figs3_1.update_layout(
                        #         margin = dict(l=5, r=5, t=5, b=5),
                        #         height = 300,
                        #         width = 300,
                        #         showlegend=False, annotations =[dict(text=f"score{score*100:.2f}%",
                        #                                             x=0.5, y=0.5,font_size=50, showarrow=False, 
                        #                                             xanchor="center")
                        #                                             ])
                        # figs3_1.update_traces(textinfo='none',marker = dict(colors=colors))
                        # st.plotly_chart(figs3_1)
                        
                        score_chart = make_donut(round(score, 2), 'Score', 'green')
                        st.altair_chart(score_chart)

                    with col1:
                        st.write('Accuracy score')
                        values3_2 = [accuracyscore,0]
                        # figs3_2 = go.Figure(data=[go.Pie(values=values3_2, hole=0.6)])
                        # figs3_2.update_layout(
                        #     margin = dict(l=5, r=5, t=5, b=5),
                        #     height = 300,
                        #     width = 300,
                        #     showlegend=False, annotations =[dict(text=f"accuracy{accuracyscore*100:.2f}%",
                        #                                         x=0.5, y=0.5,font_size=50, showarrow=False, 
                        #                                         xanchor="center")
                        #                                         ])
                        # figs3_2.update_traces(textinfo='none', marker = dict(colors=['lightgreen', 'white']))
                        # st.plotly_chart(figs3_2)
                        acc_chart = make_donut(round(accuracyscore, 2), 'Accuracy score', 'green')
                        st.altair_chart(acc_chart)
                else:                    
                    col1.write("Score error")
                
            col2_1_2.pyplot(fig)

        houghLineTransform()
else:
    col2.write('Upload image to begin!')

# Pressure score calculations    
if 'csv_data' in st.session_state and csv_data is not None:
     
    std_max = 15
    
    col3.markdown(f"""
                <div data-testid="stMetric" style="
                    background-color: #262730;
                    text-align: center;
                    padding: 15px 10px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0);
                    margin-bottom: 1rem;
                    font-family: 'Source Sans Pro', sans-serif;
                    color: white;
                ">
                    <div data-testid="stMetricLabel" style="color: #FFFFFF; font-size: 0.9rem; margin-bottom: 0.25rem;">
                        Pressure max standard deviation
                    </div>
                    <div style="
                        font-size: 1rem; 
                        font-weight: 300; 
                        color: white; 
                        background-color: #262730;
                        padding: 0; 
                        margin: 0;
                        border-radius: 0;
                    ">
                        <em>{std_max}</em>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    smoothed = pd.Series(pressure_list).rolling(window=5, center=True).mean()

    rolling_std = pd.Series(pressure_list).rolling(window=4).std()

    avg_rolling_std = rolling_std.mean()
    if pd.isna(avg_rolling_std): 
        avg_rolling_std=0

    pressurescore = max(0, 1 - avg_rolling_std/std_max)
      
    with col1:
        # values3_3 = [pressurescore,0]
        # figs3_3 = go.Figure(data=[go.Pie(values=values3_3, hole=0.6)])
        # figs3_3.update_layout(
        #                     margin = dict(l=5, r=5, t=5, b=5),
        #                     height = 300,
        #                     width = 300,
        #                     showlegend=False, annotations =[dict(text=f"pressure{pressurescore*100:.2f}%",
        #                                                         x=0.5, y=0.5,font_size=50, showarrow=False, 
        #                                                         xanchor="center")
        #                                                         ])
        # figs3_3.update_traces(textinfo='none', marker = dict(colors=['lightgreen', 'white']))
        # st.plotly_chart(figs3_3)
        st.write('Pressure score')
        press_chart = make_donut(round(pressurescore, 2), 'Pressure score', 'green')
        st.altair_chart(press_chart)
else:
    col2.write('Upload pressure data to begin!')

with col2_2_1:
    tab1, tab3, tab2 = st.tabs(["Acceleration vs Time", 'Pressure vs Time', "Acceleration vs Pressure" ])

    # time_data = np.array(acc_press_time_data['Time'])

with tab1:
    if csv_data is not None:

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
            start = start_time(dataframe)
            return time_index_to_value(dataframe, int(start + 1)) - start

        def time_value_to_index(datalist, time):
            return int(time / time_step(datalist))
        
        min_range = time_list[0]
        max_range = time_list.iloc[-1]
        step = time_step(df)

        min, max = st.slider(label = 'Time range', min_value = min_range, max_value = max_range, value = (min_range, max_range), step = step, key = 'Acceleration_slider')

        min_index = time_value_to_index(df, min)
        max_index = time_value_to_index(df, max)
        
        press_data = np.array(pressure_list[min_index:max_index])
        acc_data = np.array(acc_list[min_index:max_index])
        
        acceleration = pd.DataFrame({
            'Time': time_list[min_index:max_index],
            'Acceleration': acc_data
        })

        pressure = pd.DataFrame({
            'Time': time_list[min_index:max_index],
            'Pressure': press_data
        })

        fig = px.line(acceleration, x = 'Time', y = 'Acceleration', labels = {'x':'Time (s)', 'y':'Acceleration (ms-2)'})
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), width=400, height=250)
        st.plotly_chart(fig)

    else:
        st.write(' ')

with tab3:
    if csv_data is not None:
        time_data = np.array(acc_press_time_data['Time'])
        def time_index_to_value(dataframe, index):
            time_column = np.array(dataframe['Time'])
            return time_column[index]
        
        def last_time_index(dataframe):
            return pd.Series.last_valid_index(dataframe)
        
        def start_time(dataframe):
            return time_index_to_value(dataframe, 0)

        def end_time(dataframe):
            return time_index_to_value(dataframe,last_time_index(dataframe))

        def time_step(dataframe):
            start = start_time(dataframe)
            return time_index_to_value(dataframe, int(start + 1)) - start

        def time_value_to_index(datalist, time):
            return int(time / time_step(datalist))
        
        min_range = time_data[0]
        max_range = time_data[-1]
        step = time_step(acc_press_time_data)

        min, max = st.slider(label = 'Time range', min_value = min_range, max_value = max_range, value = (min_range, max_range), step = step, key = 'Pressure_slider')

        min_index = time_value_to_index(df, min)
        max_index = time_value_to_index(df, max)
        
        press_data = np.array(pressure_list[min_index:max_index])
        acc_data = np.array(acc_list[min_index:max_index])
        
        acceleration = pd.DataFrame({
            'Time': time_list[min_index:max_index],
            'Acceleration': acc_data
        })

        pressure = pd.DataFrame({
            'Time': time_list[min_index:max_index],
            'Pressure': press_data
        })

        fig = px.line(pressure, x = 'Time', y = 'Pressure', labels = {'x':'Time (s)', 'y':'Pressure (Pa)'})
        st.plotly_chart(fig)

    else:
        st.write(' ')

with tab2:
    if csv_data is not None:
        # acc_press_data = pd.read_csv(csv_data)
        # time_data = acc_press_data['Time']
        # acc_data = np.array(acc_press_data['Acceleration'])
        # press_data = np.array(acc_press_data['Force'])

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
        st.plotly_chart(plot(time_list, acc_list, pressure_list))

    else:
        st.write (' ')





