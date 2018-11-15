import json
import numpy as np
import cv2
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import os
import time
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt

target_dir = (r'C:\Users\km65725\Documents\trainigdata\CameraRoll')

def Training(trainingdata,resultdata):
    print("training", trainingdata)
    print("resultdata",resultdata)
    
    #np.savetxt('loga.txt', trainingdata)

    _batch_size = 100
    _mum_classes = 10
    _epochs = 20
    _validation_split=0.2

    model = Sequential()

    model.add(Dense(540, input_dim = 540, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    model.add(Dense(800, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    model.add(Dense(400, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    model.add(Dense(54, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    model.add(Dense(54, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
    
    model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mse')
    
    # model.load_weights(weights_filename)
    
    print(model.summary())
    
    classes = model.fit(trainingdata, resultdata, batch_size=_batch_size, verbose=1, epochs=_epochs, validation_split=_validation_split)
    # classes = model.fit(trainingdata, resultdata, batch_size=_batch_size, verbose=1, epochs=_epochs)

    loss, accuracy = model.evaluate(trainingdata, resultdata)
    print("\nloss:{} accuracy:{}".format(loss, accuracy))
    # print("\nloss:{}".format(loss))
    
    # モデルの保存
    f_model = './model'
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(f_model,'openpose_model.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(f_model,'openpose_model.yaml'), 'w').write(yaml_string)
    
    # 重みの保存
    print('save weights')
    model.save_weights(os.path.join(f_model,'openpose_model_weights.hdf5'))

    # 結果のプロット
    plot_history(classes)

    # classes = model.predict(openposedataset,epochs,batch_size)
    return classes

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
    

# ディレクトリ内のJsonファイルリストを返す
def GetOldJson():
    for pathnames, dirnames, filenames in os.walk(target_dir):
        for dirname in dirnames:
            print(dirname)
            target = os.path.join(pathnames,dirname,'*')
            files = [f for f in glob(target)]
            files.sort()
            yield files     # 指定ディレクトリ内の各ディレクトリ内のファイルリストを作成し返す


# トレーニングデータを作るクラス。入力 ファイルリスト、出力 トレーニングデータと正解データ
class MakeTrainingData():
    # この2つをstaticにしたかったため、json2npattayを@staticmethodにした
    trainingdata = np.zeros((0,540))
    resultdata = np.zeros((0,54))

    @staticmethod
    def json2nparray(file_list_path):
        openposedataset = np.zeros((0,54))
            
        for file_list in file_list_path:
            with open(file_list) as f:
                data = json.load(f)
            if not data['people']:# 空欄だったときにスキップ
                continue
    
            # ディレクトリ内のJsonファイルを一つにまとめる
            openposedataset = np.append(openposedataset,np.array([data['people'][0]['pose_keypoints_2d']]), axis=0)

        # トレーニングデータと正解データを作成
        openposedataset, result = MakeTrainingData.AddResultData(openposedataset)

        # 10フレごとのデータに直す
        openposedataset,result = MakeTrainingData.TrainigData(openposedataset,result)

        # 指定したたディレクトリ内のデータをトレーニングと正解データに結合
        MakeTrainingData.trainingdata = np.append(MakeTrainingData.trainingdata,openposedataset, axis = 0)
        MakeTrainingData.resultdata = np.append(MakeTrainingData.resultdata, result, axis = 0)
    
        # Staticなので別にreturnしなくても良い
        return MakeTrainingData.trainingdata, MakeTrainingData.resultdata


    def TrainigData(openposedataset, result):
        lcount = 0
        lcount2 = 0
        middledata = np.zeros((0,540))
        trainingdata = np.zeros((0,540))
        #resultdata = np.zeros((1,540))

        while True:
            middledata = np.zeros((0,540))

            for i in range(0,10,1):
                if len(openposedataset) <= (lcount + i+1):
                    break
                else:
                    # トレーニングデータ1フレーム分を取り出す
                    middledata = np.append(middledata ,openposedataset[lcount + i,:])
            if len(openposedataset) <= (lcount + i+1):
                break
            # 取り出した1フレーム分のトレーニングデータを1*540の形にして、行に追加していく
            trainingdata = np.append(trainingdata,np.reshape(middledata,(1,540)),axis = 0)
            lcount += 1
        
        # resultは加工なし
        return trainingdata, result


    def AddResultData(openposedataset):
        # 何フレーム先を正解データとするかの定数値
        kFutureFrame = 10
        # 何フレームを学習に使うかの定数値、使ってないので削除する
        kRrainingRange = 10

        trainingdata = np.zeros((0,540))
             
        resultdata = np.zeros((0,54))
        openposedataset = np.reshape(openposedataset,(-1,54))
        
        # 正解データは頭kfutureFrame分を削除したもの
        resultdata = openposedataset[kFutureFrame*2:]
        # 訓練データは後ろkfutureFrame分を削除したもの
        openposedataset = openposedataset[:len(openposedataset)-kFutureFrame]

        return openposedataset, resultdata


if __name__ == '__main__':
    # ディレクトリ内のファイルを作成
    # ファイルリストをyieldで返す
    jsonfiles = GetOldJson()

    # ディレクトリごとに出力されたfileリストから、トレーニングデータと正解データを作成する
    for i in jsonfiles:
       MakeTrainingData.json2nparray(i)
    
    # 作成したデータで学習を行う
    Training(MakeTrainingData.trainingdata,MakeTrainingData.resultdata)
    
    print("a")