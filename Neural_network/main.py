#импортируем всё из object_detection и модуль os
from ObjectDetection import *
import os

def main():
    #видео для тестирования нейросети
    videoPath ="Video_dlya_testov/crowd.mp4"
    #заранее тренированная нейронная сеть на открытом  dataset COCO
    configPath = os.path.join('Neural_network', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    #в этом файле находятся веса нейронной сети
    modelPath = os.path.join('Neural_network', 'frozen_inference_graph.pb')
    #закгрузка всем объектов или же меток
    classesPath = os.path.join('Neural_network', 'coco.names')

    vadim = Detection(videoPath, configPath, modelPath, classesPath)
    vadim.onVideo()

if __name__ == '__main__':
    main()