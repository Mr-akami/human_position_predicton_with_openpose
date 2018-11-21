import json
import numpy as np
import cv2
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import os
import time
from glob import glob
from keras.layers import Dense, Activation
from keras import optimizers
from keras import regularizers
from keras.models import Sequential, model_from_json
import matplotlib.pyplot as plt
import shutil
import subprocess



#samba上の監視対象が置かれるディレクトリ、USBカメラのときはJson,RPiのときはImageが入る
saved_samba_dir = (r'C:\Users\km65725\Documents\Visual Studio 2015\Projects\jsondir')

#Jsonファイルが置かれる場所
#RPiのとき
#saved_json_dir = (r'C:\Users\km65725\Documents\Visual Studio 2015\Projects\jsondir')
#USBカメラのとき
saved_json_dir = saved_samba_dir

# 一度読み込んだ画像の移動先
# used_img_dir = (r'')

# openpose.exeのパス
# video_to_json_openposepath = (r'./~~~~/OpenPoseDemo.exe')

# OpenPoseが画像を保存するパス
saved_image_dir = (r'aa')

# USBカメラ用Openposeパス
# 
usbcam_openpose_path = (r'/openpose.exe')
usbcam_openpose_command =  ' '.join[usbcam_openpose_path,"--write_keypoint_json",saved_jason_dir,"--write_images",saved_image_dir]

# モデルと重みの場所と名前
f_model = './model'
weights_filename = 'openpose_model_weights.hdf5'
model_filename = 'openpose_model.json'

# 黒背景画像
img = cv2.imread('original.jpg')
#img = np.zeros_like(img_)

# ファイルの更新を監視し、変更があったときに動作するクラス
class ChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        #USBカメラの時はJsonファイル、RPiのときはimageファイルを取得
        filepath = event.src_path
        filename = os.path.basename(filepath)
        #print('%s' % filename)

        #######RPiのときのみ有効########
        # openposeを動かすバッチ。Sambaにおかれた画像をJson化する
        #subprocess.run((video_to_json_openposepath,--video,filename,--write_keypoint_json,saved_jason_dir))
        #######RPiのときここまで########

        # 画像からのjsonデータを10フレームのかたまりにして入力するデータセットを作成
        openposedataset, current_gravity = Json2Nparray(GetOldestJson())

        # データセットを入力し、10frame後の動きを予測
        if openposedataset.size == 540:
            classes, future_gravity = EstimateAction.estimate(openposedataset)
            
            # current_gravity,可視化
            
            ##### WEBカメラのとき
            list_of_files = glob(saved_image_dir)
            latest_file = max(list_of_files,key=os.path.getctime)
            latest_file_path = os.path.join(saved_image_dir,latest_file)
            img = cv2.imread(latest_file_path) 
            ##### WEBカメラのときここまで
            
            #### RPiのとき
            #img = cv2.imread(filename) 
            ### RPiのときここまで
            
            cv2.circle(img,(int(current_gravity[0]), int(current_gravity[1])),20,(255,0,255),thickness = -1)
            cv2.circle(img,(int(future_gravity[0]), int(future_gravity[1])),20,(255,0,0),thickness = -1)
            #cv2.destroyAllWindows()

            cv2.imshow('nacho', img)
            cv2.waitKey(1)
            #time.sleep(1)


        ########RPiのときのみ有効##########
        # 一度読み込んだ画像を移動する
        #shutil.move(filename,used_img_dir)


    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを変更しました' % filename)

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sを削除しました' % filename)


# DeepLearningにより、Trainingで指定したフレーム数後の状態を推定
class EstimateAction:

    @staticmethod
    def estimate(openposedataset):
        openposedataset = openposedataset.reshape((1,540))
        epochs = 20
        batch_size = 100
        # モデルの読み込み
        json_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_json(json_string)

        model.summary()

        model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

        # 重みの読み込み
        model.load_weights(os.path.join(f_model,weights_filename))

        # print(model.summary())

        classes = model.predict(openposedataset,epochs,batch_size)

        print(classes)

        future_gravity = EstimateAction.XYPosition(classes)
        print(future_gravity)

        # classes:予測した体54点のデータ、future_gravity:体の重心位置
        return classes, future_gravity


    def XYPosition(classes):
        ar = np.reshape(classes,(-1,3))
        ar_sum = ar.sum(axis = 0)
        future_gravity = (ar_sum[0]/ar.shape[0],ar_sum[1]/ar.shape[0])
        return future_gravity


def GetOldestJson():
    target = os.path.join(saved_jason_dir, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files:files[1])
    print(latest_modified_file_path[0])#[0]:oldest
    print(latest_modified_file_path[-1])#[-1]:latest
    return latest_modified_file_path[-10:]


def Json2Nparray(file_list_path):
    trainingdata = np.zeros((0,540))

    if len(file_list_path) < 10:
        # ファイルが１０個たまるまではトレーニングデータ生成を行わない
        return np.zeros(0), np.zeros(0)

    elif len(file_list_path) > 9:
        for i,file_list in enumerate(file_list_path):
            # 最新の１０ファイル分だけを予測データとする
            if i <10:
                with open(file_list[0]) as f:
                    data = json.load(f)
                trainingdata = np.append(trainingdata,np.array([data['people'][0]['pose_keypoints_2d']]))
                print('trainingdata.size=',trainingdata.size)
            if i == 9:
                ar = np.reshape(data['people'][0]['pose_keypoints_2d'],(-1,3))
                ar_sum = ar.sum(axis = 0)
                current_gragity = (ar_sum[0]/ar.shape[0],ar_sum[1]/ar.shape[0])
                return trainingdata, current_gragity
  

if __name__ == '__main__':
    print("Reday")

    #######################################
    # Rpiから画像を受け取る場合
    #######################################
    while 1:
        event_handler = ChangeHandler()
        
        observer = Observer()
        # 今はjsonファイルを格納するsaved_jason_dirが監視対象だが、本番ではimg_dirになる
        observer.schedule(event_handler, saved_jason_dir, recursive=True)#監視フォルダに画像をおく
        observer.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    ####RPiの場合ここまで#####################


    ########################################
    ## USBカメラで処理する場合
    ########################################
    ## USBカメラ起動&Jsonファイル保存
    #subprocess.Popen(usbcam_openpose_command,shell=True)# 監視フォルダにJsonファイルを置く
    ## Observer起動
    #    while 1:
    #    event_handler = ChangeHandler()
    #    
    #    observer = Observer()
    #    # jsonファイルを格納するsaved_jason_dirを指定
    #    observer.schedule(event_handler, saved_jason_dir, recursive=True)
    #    observer.start()
    #    try:
    #        while True:
    #            time.sleep(0.1)
    #    except KeyboardInterrupt:
    #        observer.stop()
    #   observer.join()
    ##USBカメラの場合ここまで##################


