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
# img = cv.imread(line_image, cv.IMREAD_GRAYSCALE)
# st.pyplot(img)

# col1, col2 = st.columns(4,5)
# st.container()
row1 = st.columns([0.4,0.6])
row2 = st.columns(1)

# for col in row1 + row2:
#     tile = col.container(height=120)
con1 = row1[0].container()
with con1:
    st.subheader('Original line image')
    con1_1 = st.container(height = 350, border=True)


con2 = row1[1].container()
with con2:
    st.subheader('Transform output image')
    con2_1 = st.container(height = 350, border=True)



with row2[0]:
    st.subheader("Line equations")
if line_image:
    with tempfile.NamedTemporaryFile(delete=False,suffix = '.jpg') as temp_image:
        temp_image.write(line_image.getbuffer())
        temp_image.seek(0)
        # image_path = temp_image.name

        # original_image = temp_image.read()
        con1_1.image(temp_image.name)
        # st.image(line_image)
        def houghLineTransform():
            img = cv.imread(temp_image.name, cv.IMREAD_GRAYSCALE)
            if img is None:
                st.write('File read unsuccesful. Please check file type.')
 
            imgBlur = cv.GaussianBlur(img, (21,21), 3)
            cannyEdge =cv.Canny(imgBlur, 50, 180)

            fig = plt.figure(figsize=(18,10))
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)

            plt.imshow(cannyEdge)
            # plt.text(-50, 500, "Global Text", fontsize=14, color='red', ha='center')

            distResol = 1
            angleResol = np.pi/180
            threshold = 150
            lines = cv.HoughLines(cannyEdge,distResol, angleResol, threshold)
            k=3000

            for curline in lines:
                rho,theta = curline[0]
                dhat = np.array([[np.cos(theta)],[np.sin(theta)]])
                d = rho*dhat
                lhat = np.array([[-np.sin(theta)],[np.cos(theta)]])
                p1 = d + k*lhat
                p2 = d - k*lhat
                p1 = p1.astype(int)
                p2 = p2.astype(int)
                cv.line(img, (p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (255,255,255), 10)

                # Line drawn 
                # ''' xcos(theta) + ysin(theta) = p
                #     y = -cos(theta)/sin(theta)x + p/sin(theta)
                
                # '''
                if np.sin(theta) != 0:
                    m = -np.cos(theta)/np.sin(theta)
                    b = rho/np.cos(theta)

                if m:
                    row2[0].write(f"Line equation: y = {m:.2f}x + {b:.2f}")

            con2_1.pyplot(fig)

        houghLineTransform()