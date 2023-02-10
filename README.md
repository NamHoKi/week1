# object detection toy project

<hr>

## Step

### 1. Dataset
### 2. Train
### 3. Inference
### 4. GUI
### 5. Reference


<hr>

## 1. Dataset
### Label
```
label_dict = {
    0: 'belt',
    1: 'no_belt',
    2: 'hoes',
    3: 'no_shoes',
    4: 'helmet',
    5: 'no_helmet',
    6: 'person'
}
```

### Image
[AI hub 공사장 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=163)

[안전모](https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/4/images/?split=train)

[안전모](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/browse?queryText=class%3ANO-Hardhat&pageSize=50&startingIndex=500&browseQuery=true)

[안전모](https://public.roboflow.com/object-detection/hard-hat-workers)

[신발](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=163)

[옷](https://universe.roboflow.com/yamin-thwe-weurg/e-commerce-puyv6/browse?queryText=&pageSize=50&startingIndex=300&browseQuery=true)

[옷](https://universe.roboflow.com/zhang-ya-ying/clothes-detect-fevqm/browse?queryText=&pageSize=50&startingIndex=150&browseQuery=true)


### Person Label
```
Yolov5 - detect.py - pretrained model (yolo5s, default) - label:0
```

<hr>

## 2. Train
### [Yolo v5](https://github.com/ultralytics/yolov5)
### Pytorch version
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<hr>

## 4. GUI

```
# PyQt5 install
pip install pyqt5
pip install pyside2
```

<hr>

## 5. Reference

### https://koreascience.kr/article/JAKO201915555313326.pdf
