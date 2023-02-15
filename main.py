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

# 使用遞迴迴圈尋找所有子目錄內中的檔案，將其轉換為字典格式
def get_node(path):
    nodes = []
    for folder in os.listdir(path):
        node = {}
        node["label"] = folder
        node["value"] = os.path.join(path,folder)
        if os.path.splitext(folder)[-1] not in image_type + video_type + ['']:
            node["disabled"] = True
        sub_path = os.path.join(path,folder)
        if os.path.isdir(sub_path):
            node["children"] = get_node(sub_path)
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

def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img

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
    check_path(os.path.join('temp',user,'frames'))
    with layout:
        for file in stqdm(files):
            vidcap = cv2.VideoCapture(os.path.join("temp",user,"video",file))
            success,image = vidcap.read()
            fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps
            for i in stqdm(range(int(duration*set_fps))):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,i*(1/set_fps))
                success,image = vidcap.read()
                if not success:
                    layout.warning(f'Fail to get frame from `{file}` in `{i*(1/set_fps)}` sec')
                    continue
                name = f"{os.path.splitext(file)[0]}_{i:04d}".replace('.','')
                file_name = os.path.join('temp',user,'frames',name)
                sss.result[file_name] = {}
                PIL_image = Image.fromarray(image[:,:,::-1])
                sss.result[file_name]['image_str'] = str(image_to_base64(PIL_image))[2:-1]
                cv2.imwrite(file_name+'.png',image)

def api(img_base64,caption):
    r = requests.post('http://127.0.0.1:5000/upload', 
    data = {
        'base64_str':f'{img_base64}',
        'caption': caption
    })

    return json.loads(r.text)

def xywh2xyxy(x):  # 将中心坐标和高宽，转成左上角右下角的坐标
        y = x.new(x.shape)  # 创建一个新的空间
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

def plot_output(base64_str,output):
    im = base64_to_image(base64_str)
    im0 = np.array(im)
    annotator = Annotator(im0, line_width=3)
    h,w ,_ = im0.shape
    for cls in range(len(keys(output))):
        box = output[str(cls)]
        c = box['label']
        p = round(box['prob'],2)
        xyxy = xywh2xyxy(torch.Tensor([box['x']*w,box['y']*h,box['w']*w,box['h']*h]))
        label = f'{c} {p}'
        annotator.box_label(xyxy, label, color=colors(cls, True))
    image = Image.fromarray(annotator.result())
    return image

def zipdir(path, ziph):
    # ziph is zipfile handle
    zip_layout = st.empty()
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in ['.txt','.png','.names','.data']:
                zip_layout.write(f'`{file}`')
                ziph.write(os.path.join(root, file), 
                        os.path.relpath(os.path.join(root, file),os.path.join(path)))
    zip_layout.empty()


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
            select_files = [file for file in tree_select(get_node('/mnt/share/share'))['checked'] if os.path.isfile(file)]

        for select_file in select_files:
            select_file_fold, select_file_name = os.path.split(select_file)
            make_path = os.path.join('temp',username,os.path.split(select_file_fold)[-1])
            check_path(make_path)
            shutil.copyfile(select_file,os.path.join(make_path,select_file_name))


        # 遍歷temp資料夾中的所有檔案，並刪除不需要的檔案，並將圖像和視頻檔案存儲在分別的列表中
        image_files, video_files = [], []
        path = os.path.join('temp', username)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if file in [os.path.basename(n) for n in select_files] or root.find('frame') >= 0:
                    if os.path.splitext(file)[1] in image_type:
                        image_files.append(file_path)
                    elif os.path.splitext(file)[1] in video_type:
                        video_files.append(file_path)
                elif os.path.isfile(file_path):
                    os.remove(file_path)


        all_names = [re.sub('[\u4e00-\u9fa5]', '', os.path.splitext(f)[0]).replace(' ', '_') for f in image_files + video_files]
        
        # 檢查圖像檔案的名稱是否包含中文字符，如果包含中文字符，則發出警告。
        zh_warning = sum(1 for split_name in image_files if u'\u4E00' <= split_name <= u'\u9FFF')
        if zh_warning > 0:
            letf1_expander.warning("使用中文檔名可能會造成錯誤", icon="⚠️")


        # - 將圖像檔轉換為 base64 編碼，並將結果存儲在 sss.result 字典中。
        # - 將視頻檔重命名，並在應用程序畫面中顯示視頻。
        # - 如果存在視頻檔，則獲取視頻的相關信息（最大帧率和總持續時間），並根據用戶選擇的帧率進行影片轉圖像的操作。

        for uploaded_image in stqdm(image_files, total=len(image_files), desc="Processing images"):
            file_name = re.sub('[\u4e00-\u9fff]', '', os.path.splitext(uploaded_image)[0].replace(' ', '_').replace('.', ''))
            if file_name not in keys(sss.result):
                sss.result[file_name] = {}
                PIL_image = Image.open(uploaded_image, mode='r')
                sss.result[file_name]['image_str'] = str(image_to_base64(PIL_image))[2:-1]

        for uploaded_video in video_files:
            file_name = re.sub('[\u4e00-\u9fff]', '', os.path.basename(uploaded_video).replace(' ', '_'))
            video_path = os.path.join('temp', username, 'video', file_name)
            check_path(os.path.join('temp', username, 'video'))
            if not os.path.isfile(video_path):
                os.rename(uploaded_video, video_path)
            right1_expander.markdown(f'### {file_name}')
            if file_name.split('.')[-1] != '.mp4':
                right1_expander.warning('MP4 以外格式可能無法預覽')
            right1_expander.video(video_path)

        if video_files:
            max_fps, total_duration = get_info()
            set_fps = Set_fps(max_fps, total_duration, letf2_expander)
            if letf2_expander.button("選擇影片偵率", type='primary'):
                video2image(set_fps, letf2_expander)


        # - 單擊開始按鈕時，代碼遍歷sss.result對象的所有鍵並調用 GLIP API 對每個圖像執行對象檢測。返回的輸出存儲在sss.outputs對像中。
        # - 如果選中測試模式複選框，則代碼僅繪製第一張圖像的原始圖像和註釋圖像的比較圖。
        # - 如果未選中測試模式複選框，則代碼會為對像中的每個圖像繪製帶註釋的圖像sss.result。註釋圖像顯示在 Streamlit 應用程序中，
        # - 如果過程成功，則會顯示成功消息“FOXY 成功標記記所有圖像！”。

        if len(keys(sss.result)) > 0:
            with right3_expander:
                st.markdown('### 描述需要檢測的對象')
                caption = st.text_input('僅支援英文輸入','cone . a man . hard hat.')
                button_layout = st.columns([1,1,1,3])
                image_area = st.empty()
                # sss.prob = button_layout[0].slider('confidence threshold', 0.0, 1.0, 0.5)
                compare = button_layout[2].checkbox('測試模式',value=True)
                if video_files : sss.Pass = button_layout[3].checkbox('跳過標注（僅輸出圖片）',value=False)
                if button_layout[1].button("重置",type='primary'):
                    sss.outputs = {}
                    st.info('🔄 重置成功')
                    
                if button_layout[0].button('開始', type='primary'):
                    for e, file_name in enumerate(stqdm(keys(sss.result))):
                        if sss.Pass:
                            continue
                        try:
                            output = api(sss.result[file_name]['image_str'], caption)
                        except Exception as e:
                            st.error('🚨 sorry some thing error in connect GLIP API')
                            st.write(e)
                        if 'message' in keys(output):
                            st.write(output['message'])
                            st.write(f'🚨 sorry some thing error in {file_name}')
                            continue
                        if file_name not in keys(sss.outputs):
                            sss.outputs[file_name] = output
                        else:
                            n = int(keys(sss.outputs[file_name])[-1]) + 1 if len(keys(sss.outputs[file_name])) != 0 else 0
                            for k in range(len(keys(output))):
                                sss.outputs[file_name][str(k + n)] = output[str(k)]
                        image_plt = plot_output(sss.result[file_name]['image_str'], sss.outputs[file_name])
                        image_org = base64_to_image(sss.result[file_name]['image_str'])
                        sss.size[file_name] = {'w': image_org.size[0], 'h': image_org.size[1]}

                        with image_area:
                            if compare:
                                image_comparison(image_plt,image_org)
                                break
                            else :
                                st.image(image_plt,caption=caption) 

                        for j in range(5):
                            if e%5 == j: 
                                with view_layout[j] : 
                                    st.image(image_plt,caption=file_name[-4:])
                    if compare: 
                        st.warning('⚠️ 在測試模式下只標記第一張圖片')
                    else: 
                        st.success('😊 FOXY 成功標記所有圖像！')
    
    # 創建一個字典，並在側邊欄中顯示分類標籤，並提供選擇數量和信心閾值的選項。字典的鍵為分類標籤，值是兩個選擇框和一個滑塊。
    dic = {}
    with st.sidebar:
        if len(keys(sss.result)) > 0:
            all_label = []
            for img_path in keys(sss.outputs):
                for cls in keys(sss.outputs[img_path]):
                    label = sss.outputs[img_path][cls]['label']
                    if label not in all_label:
                        all_label.append(label)

            for label in all_label:
                st.markdown(f'### {label}')
                sidebar_layout = st.columns([1.2, 2])
                num_selected = sidebar_layout[0].selectbox(f'number', range(50),key=label+'selectbox')
                prob_threshold = sidebar_layout[1].slider(f'confidence threshold', 0.5, 0.9, 0.5,key=label+'slider')
                dic[label] = {
                    'label': num_selected,
                    'prob': prob_threshold
                }
                for img_path in keys(sss.outputs):
                    for cls in keys(sss.outputs[img_path]):
                        label_ = sss.outputs[img_path][cls]['label']
                        if label_ == label:
                            sss.outputs[img_path][cls]['index'] = num_selected


        # 移除outputs中機率低於設定prob的物件
        if len(keys(sss.outputs))>0 or sss.Pass:
            if st.button('儲存',type='primary') or sss.Pass:

                for label in all_label:
                    for img_path in keys(sss.outputs):
                        for cls in keys(sss.outputs[img_path]):
                            if sss.outputs[img_path][cls]['prob'] < dic[label]['prob']:
                                sss.outputs[img_path].pop(cls)

                with open('./lottie/lf30_editor_3p1jeacf.json','r') as f:
                    lottie_json = json.loads(f.read())

                st_lottie(lottie_json,height=200,width=200)
                ept2 = st.empty()
                ept2.markdown('### wait a moment 🚀')
                if not os.path.isdir(os.path.join('temp',user,'download')):
                        os.makedirs(os.path.join('temp',user,'download'))


                # - 創建并寫入文件 train.txt ，其中保存了每個圖像的路徑信息。
                # - 對於每個圖像，創建一個同名的文本文件，其中保存了圖像中每個物體的座標信息。
                # - 創建文件 obj.data ，保存了一些模型訓練所需的配置信息，包括類別數量、訓練集路徑、物體名稱以及备份文件夾。
                # - 創建文件 obj.names ，保存了物體名稱的列表。
                for root, dirs, files in os.walk(os.path.join('temp', username)):
                    for file in files:
                        if os.path.isfile(os.path.join(root, file)) and os.path.splitext(file)[1] in image_type:
                            if not os.path.isfile(f'temp/{user}/download/{file}'):
                                os.rename(
                                    os.path.join(root, file),
                                    f'temp/{user}/download/{file}'
                                )

                with open(os.path.join('temp', user, 'train.txt'), 'w') as f1:
                    for img_path in keys(sss.outputs):
                        output_path = os.path.join('data', 'download', f'{os.path.basename(img_path)}.png')
                        f1.write(output_path + '\n')
                        with open(os.path.join('temp', username, 'download', f'{os.path.basename(img_path)}.txt'), 'w') as f:
                            for cls in keys(sss.outputs[img_path]):
                                w = sss.size[img_path]['w']
                                h = sss.size[img_path]['h']
                                box = sss.outputs[img_path][cls]
                                xyxy = xywh2xyxy(torch.Tensor([box['x'] * w, box['y'] * h, box['w'] * w, box['h'] * h]))
                                f.write('{} {} {} {} {}\n'.format(box['index'], box['x'], box['y'], box['w'], box['h']))

                with open(os.path.join('temp', user, 'obj.data'), 'w') as f2:
                    f2.write('classes = 6\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/')

                with open(os.path.join('temp', user, 'obj.names'), 'w') as f3:
                    f3.write('illegal\nsafety\ncone\ntruck\nchemical\nfilling')


                #將用戶生成的文件夾進行打包壓縮，並提供下載按鈕
                dts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with zipfile.ZipFile(f'zip/{user}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipdir(f'temp/{user}', zipf)

                # copy_path = os.path.joint('/mnt','share','share','FOXY',user)
                # check_path(copy_path)
                # shutil.copyfile(f'zip/{user}.zip',os.path.join(copy_path,f"{dts}.zip"))
                ept2.markdown('### success ☺️')

                with open(f'zip/{user}.zip', "rb") as file:
                    btn = st.download_button(
                            label="Download",
                            data=file,
                            file_name=f"{dts}.zip",
                        )
