import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Adding logo and configuring page layout

logo = 'https://github.com/nahyunnnn/payload-25/blob/main/github_validation/pages/icons/online_ares_logo.jpeg?raw=true'
st.set_page_config(layout='wide', page_title = 'Simulation of Payload Tetra', page_icon=logo)
# st.logo(logo, size = 'large')

st.title('Simulation of Payload Tetra')
def run():
    iframe_src = "https://www.desmos.com/3d/j2bhvgzpr6"
    components.iframe(iframe_src, height = 450)

if __name__ == "__main__":
    run()

st.markdown("Go to 'Coords' and press play on 'S' to watch Tetra move")  # can also use st.caption but not as clear
