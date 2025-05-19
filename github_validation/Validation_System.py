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
import math

# Adding logo and configuring page layout
logo = 'https://github.com/nahyunnnn/payload-25/blob/main/github_validation/pages/icons/online_ares_logo.jpeg?raw=true'
st.set_page_config(layout='wide', page_title='Main', page_icon=logo)
st.logo(logo, size = 'large')

row00 = st.columns([2,8])
row00[1].title('Lemaire Payload Line Validation', )

row0 = st.columns([1,1])
line_image = row0[0].file_uploader('Upload line image')
pressure_data = row0[1].file_uploader('Upload pressure data')
st.write("---")

####################################
#       Container formatting       #
#      ______________________      #
# row1 |  con1_1 | con1_2   |      #
#      |_________|__________|      #
# row2 |   con2_1    |con2_2|      #
#      |_____________|______|      #
# row3 |con3_1|con3_2|con3_3|      #
#      |______|______|______|      #
####################################

row1 = st.columns([0.4,0.6])
row2 = st.columns([2, 1])
row3 = st.columns(3)

con1_1 = row1[0].container()
with con1_1:
    st.subheader('Original line image')
    con1_1 = st.container(height = 350, border=True)

con1_2 = row1[1].container()
with con1_2:
    st.subheader('Transform output image')
    con1_2_1 = st.container(height = 350, border=True)

con2_1 = row2[0].container(border=True)
with con2_1:
    st.subheader("Line of Best Fit")

con2_2 = row2[1].container(border=True)
with con2_2:
    st.subheader("Max std for pressure")

con3_1 = row3[0].container(border = True)
with con3_1:
    st.subheader('Straightness')

con3_2 = row3[1].container(border = True)
with con3_2:
    st.subheader('Accuracy')

con3_3 = row3[2].container(border = True)
with con3_3:
    st.subheader('Pressure')

# Validation system code    
if line_image:
    with tempfile.NamedTemporaryFile(delete=False,suffix = '.jpg') as temp_image:
        temp_image.write(line_image.getbuffer())
        temp_image.seek(0)
        con1_1.image(temp_image.name)
    img = cv.imread(temp_image.name, cv.IMREAD_GRAYSCALE)
    if img is None:
        st.write('File read unsuccesful. Please check file type.')
 
    imgBlur = cv.GaussianBlur(img, (15,15), 3)
    thresholded_image = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)        
    cannyEdge =cv.Canny(imgBlur, 50, 180)
    
    def calculate_straightness(edge_image, tolerance):
        edge_points = np.column_stack(np.where(edge_image > 0))
        if len(edge_points) < 2:
            return None
        
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

        con2_1.latex(f"\Large y={m:.2f}x+{b:.2f}")

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
            con3_1.write("Hough Lines detected,") #Score automatically 1 
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
                figs3_1 = go.Figure(data=[go.Pie(values=values3_1, hole=0.6)])
                figs3_1.update_layout(showlegend=False,
                                      annotations =[dict(text=f"{100*score:.2f}%",x=0.5, y=0.5,
                      font_size=50, showarrow=False, xanchor="center")])
                figs3_1.update_traces(textinfo='none',marker = dict(colors=colors))
                con3_1.plotly_chart(figs3_1)

                with con3_2:
                    values3_2 = [accuracyscore,0]
                    figs3_2 = go.Figure(data=[go.Pie(values=values3_2, hole=0.6)])
                    figs3_2.update_layout(
                        showlegend=False, annotations =[dict(text=f"{accuracyscore*100:.2f}%",
                                                            x=0.5, y=0.5,font_size=50, showarrow=False, 
                                                            xanchor="center")
                                                            ])
                    figs3_2.update_traces(textinfo='none', marker = dict(colors=['lightgreen', 'white']))
                    con3_2.plotly_chart(figs3_2)
            else:                    
                con3_1.write("Score error")
            
        con1_2_1.pyplot(fig)

    houghLineTransform()
else:
    con3_1.write('Upload image to begin!')
    con3_2.write('Upload image to begin!')

# Pressure score calculations    
if pressure_data is not None:
     
    std_max = 15
    con2_2.latex(f'\large Standard\: deviation = {std_max}')

    reader  = pd.read_csv(pressure_data)
    pressure_list = reader['Force']
    time_list = reader['Time']

    smoothed = pd.Series(pressure_list).rolling(window=5, center=True).mean()

    rolling_std = pd.Series(pressure_list).rolling(window=4).std()

    avg_rolling_std = rolling_std.mean()
    if pd.isna(avg_rolling_std): 
        avg_rolling_std=0

    pressurescore = max(0, 1 - avg_rolling_std/std_max)
      
    with con3_3:
        values3_3 = [pressurescore,0]
        figs3_3 = go.Figure(data=[go.Pie(values=values3_3, hole=0.6)])
        figs3_3.update_layout(showlegend=False, annotations =[dict(text=f'{pressurescore*100:.2f}%',x=0.5, y=0.5,
                            font_size=50, showarrow=False, xanchor="center")])
        figs3_3.update_traces(textinfo='none', marker = dict(colors=['lightgreen', 'white']))
        con3_3.plotly_chart(figs3_3)
else:
    con3_3.write('Upload pressure data to begin!')
    
    
    



