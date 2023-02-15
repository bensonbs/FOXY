#!/bin/bash
cd FOXY
streamlit run main.py &
cd ~/../workspace/FOXY/GLIP/
python main.py&
cd ~/../workspace/
jupyter lab
