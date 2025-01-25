import streamlit as st
import altair as alt
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tempfile

st.set_page_config(layout='wide', page_title='Main')

row00 = st.columns([2,8])
row00[1].title('Lemaire Payload Line Validation', )

row0 = st.columns([2,5,2])
line_image = row0[1].file_uploader('Upload line image')

st.write("---")

row1 = st.columns([0.4,0.6])
row2 = st.columns(1)
row3 = st.columns(3)

con1 = row1[0].container()
with con1:
    st.subheader('Original line image')
    con1_1 = st.container(height = 350, border=True)

con1_2 = row1[1].container()
with con1_2:
    st.subheader('Transform output image')
    con1_2_1 = st.container(height = 350, border=True)

con2 = row2.container

con3_1 = row3[0].container()
con3_3 = row3[1].container()

with row2[0]:
    st.subheader("Line equations")

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
        con2.write(f"Line of Best Fit: y={m:.2f}x+{b:.2f}")
        x_min, x_max = int(np.min(x)), int(np.max(x))
        y_min, y_max = int(m*x_min + b), int(m*x_max+b)
        cv.line(img, (x_min, y_min), (x_max, y_max), (0,0,255), 2)

        score = 1-np.mean(normal_distances)
        return score

    def houghLineTransform():

        fig = plt.figure(figsize=(18,10))
        plt.subplot(121)
        plt.imshow(img)            
        plt.subplot(122)

        plt.imshow(cannyEdge)
        # plt.text(-50, 500, "Global Text", fontsize=14, color='red', ha='center')

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
                    con3_1.write(f"Line Equation: y = {m:.2f}x+{b:.2}")
            con3_1.write("Hough Lines detected,") #Score automatically 1 
        else:
            score = calculate_straightness(cannyEdge, tolerance = 100)
            if score is not None:                    
                con3_3.write(f"Straightness Score: {score:.2f}")
            else:                    
                con3_3.write("gg")
            
        con2_1.pyplot(fig)

    houghLineTransform()