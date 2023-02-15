import os,io,re,cv2,requests,json,base64,torch,zipfile,datetime,yaml
import streamlit as st
import numpy as np
import shutil
import streamlit_nested_layout
import streamlit_authenticator as stauth
from streamlit import session_state as sss
from PIL import Image
from stqdm import stqdm
from pathlib import Path
from utils.plots import Annotator, colors
from streamlit_image_comparison import image_comparison
from streamlit_lottie import st_lottie
from streamlit_tree_select import tree_select


def keys(dic): return list(dic.keys())

def st_init(var,default): 
    if var not in sss: sss[var] = default

def get_node(path):
    nodes = []
    folders = os.listdir(path)
    for folder in folders:
        node = {}
        node["label"] = folder
        node["value"] = os.path.join(path,folder)
        sub_path = os.path.join(path,folder)
        if os.path.isdir(sub_path):
            node["children"] = []
            for folder2 in  os.listdir(sub_path):
                node2 = {}
                node2["label"] = folder2
                node2["value"] = os.path.join(sub_path,folder2)
                sub_path2 = os.path.join(path,folder,folder2)
                if os.path.isdir(sub_path2):
                    node2["children"] = []
                    for folder3 in  os.listdir(sub_path2):
                        node3 = {}
                        node3["label"] = folder3
                        node3["value"] = os.path.join(sub_path2,folder3)
                        sub_path3 = os.path.join(path,folder,folder2,folder3)
                        if os.path.isdir(sub_path3):
                            node3["children"] = []
                            for folder4 in  os.listdir(sub_path3):
                                node4 = {}
                                node4["label"] = folder4
                                node4["value"] = os.path.join(sub_path3,folder4)
                                
                                node3["children"].append(node4)
                        node2["children"].append(node3)
                node["children"].append(node2)
        nodes.append(node)
    return nodes

def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def image_to_base64(img):
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def get_info():
    max_fps = 60
    total_duration = 0
    files = os.listdir(os.path.join("temp",user,"video"))
    for file in files:
        vidcap = cv2.VideoCapture(os.path.join("temp",user,"video",file))
        success,image = vidcap.read()
        if success:
            fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
            if fps < max_fps:
                max_fps = fps
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration += frame_count/fps
            
        else:
            st.write('some thing error in load video')

    return max_fps,total_duration

def Set_fps(fps,duration,layout):
    with layout:
        st.markdown('### 我們應該多久採樣一次這個視頻？')
        set_fps = st.slider('Select a range of values',min_value=1/fps, max_value=10.0, value=1.0, step=0.1,format='1 fram every %.2f seconds')
        set_fps = 1/set_fps

        # st.markdown(f'{round(set_fps,2)} fram / seconds' if set_fps > 1 else f'### 1 fram every {round(1/set_fps,2)} seconds')
        st.markdown(f'### 輸出數量: `{int(duration*set_fps)}` 張圖')
        return set_fps

def video2image(set_fps,layout):
    files = os.listdir(os.path.join("temp",user,"video"))
    with layout:
        for file in stqdm(files):
            st.write(os.path.join("temp",user,"video",file))
            vidcap = cv2.VideoCapture(os.path.join("temp",user,"video",file))
            success,image = vidcap.read()
            fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps
            for i in stqdm(range(int(duration*set_fps))):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,i*1000*(1/set_fps))
                success,image = vidcap.read()
                if not success:
                    layout.warning(f'Fail to get frame from `{file}` in `{i*(1/set_fps)}` sec')
                    continue
                file_name = f"{os.path.splitext(file)[0]}_{i:04d}"
                sss.result[file_name] = {}
                PIL_image = Image.fromarray(image[:,:,::-1])
                sss.result[file_name]['image_str'] = str(image_to_base64(PIL_image))[2:-1]
                cv2.imwrite(os.path.join('temp',user,'frames',f"{file_name}.png"),image)

if __name__ == '__main__':

    st.set_page_config(layout="wide")

    st_init(var='result', default={})
    st_init(var='outputs',default={})
    st_init(var='size',default={})
    st_init(var='prob',default=0.5)
    st_init(var='Pass',default=False)
    uploaded_image=False

    image_type = [".png",".jpg"]
    video_type = [".mp4",".avi",".mkv",'.asf']

    layout = st.columns(2)
    letf1_expander = layout[0].expander('file uploader', expanded=True)
    right1_expander = layout[0].expander('影片預覽', expanded=False)
    right2_expander = layout[0].expander('圖片預覽', expanded=False)
    letf2_expander = layout[0].container()
    right3_expander = layout[1]
    view_layout = st.columns(5)

    with open('./config.yaml') as file:
        config = yaml.load(file)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    with st.sidebar:
        st.image('pic/foxy.png')
        st.image('pic/foxy_icon.jpg')
        user, authentication_status, username = authenticator.login('Login', 'main')

        if authentication_status:
            if os.path.isfile(f'zip/{user}.zip'): os.remove(f'zip/{user}.zip')
            st.write(f'歡迎 *{user}*')
            authenticator.logout('登出', 'main')
            if not os.path.isdir(f'temp/{user}'):
                os.makedirs(f'temp/{user}')
        elif authentication_status == False:
            st.error('帳號/密碼錯誤')
        elif authentication_status == None:
            st.warning('請輸入帳號/密碼')

    
    if (not authentication_status) or (len(keys(sss.result))==0):
        uploaded_image=False
        #clear chace
        sss.result={}
        sss.outputs={}
        #remove temp
        for root, dirs, files in os.walk(f'temp/{user}', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    if authentication_status:
        with letf1_expander:
            st.markdown('### 上傳圖片/視頻文件')
            select_files = [file for file in tree_select(get_node('upload'))['checked'] if os.path.isfile(file)]
        for uploaded_file in select_files:
            temp_path = os.path.join('temp',username)
            check_path(temp_path)
            shutil.copyfile(uploaded_file, os.path.join(temp_path,os.path.basename(uploaded_file)))
        
            uploaded_files = os.listdir(temp_path)
            image_files = [v for v in uploaded_files if os.path.splitext(v)[1] in image_type]
            video_files = [v for v in uploaded_files if os.path.splitext(v)[1] in video_type]
            all_names   = [re.sub('[\u4e00-\u9fa5]','',os.path.splitext(uploaded_image)[0]).replace(' ','_') for uploaded_image in uploaded_files]

            #中文偵測
            zh_warning = 0
            for split_name in image_files:
                if u'\u4E00' <= split_name <= u'\u9FFF':
                    zh_warning += 1
            if zh_warning > 0:
                letf1_expander.warning('使用中文檔名可能會造成錯誤', icon="⚠️")

            for e,uploaded_image in stqdm(enumerate(image_files)):
                file_name = os.path.splitext(uploaded_image)[0].replace(' ','_')
                file_name = re.sub('[\u4e00-\u9fff]','',file_name)
                sss.result[file_name] = {}
                fh = os.path.join('temp',username,uploaded_image)
                img = Image.open(fh, mode='r')
                sss.result[file_name]['image_str'] = image_to_base64(img)

            for uploaded_video in video_files:
                file_name = os.path.basename(uploaded_video).replace(' ','_')
                file_name = re.sub('[\u4e00-\u9fff]','',file_name)
                check_path(os.path.join('temp',username,'video'))
                os.rename(os.path.join('temp',username,uploaded_video),os.path.join('temp',username,'video',file_name))
                right1_expander.markdown(f'### {file_name}')
                if file_name.split('.')[-1] != '.mp4':
                    right1_expander.warning('MP4 以外格式可能無法預覽')
                right1_expander.video(os.path.join('temp',username,'video',file_name))

            if video_files:
                max_fps,total_duration = get_info()
                set_fps = Set_fps(max_fps,total_duration,letf2_expander)
                if letf2_expander.button('選擇影片偵率',type='primary'):
                    video2image(set_fps,letf2_expander)