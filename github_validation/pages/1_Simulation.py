import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout='wide', page_title = 'Simulation of Payload Tetra')
def run():
    iframe_src = "https://www.desmos.com/3d/j2bhvgzpr6"
    components.iframe(iframe_src, height = 450)

if __name__ == "__main__":
    run()
