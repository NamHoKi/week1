# object detection toy project
1. 공사 현장 안전 장비 미착용으로 인한 사고 방지 (스마트 안전 통합 관제 시스템)
2. 배운 내용으로 데이터 셋 구축 및 서비스 구현

<참고자료>

http://news.kmib.co.kr/article/view.asp?arcid=0016070557&code=61121111&cp=nv

https://newsis.com/view/?id=NISX20230127_0002172017

https://www.hankyung.com/society/article/202302032530Y


<hr>
-*- encoding - python 3.8 -*-

<hr>

## Step

### 1. Dataset
데이터 다운로드 및 데이터 선택
이미지 검수 및 이미지 제거 또는 CVAT로 수정작업
### 2. Train
yolov5모델을 선택하여 training
### 3. Inference
### 4. GUI
pyqt5로 GUI, 추가 기능 구현
### 5. Reference


<hr>

## 1. Dataset
### Label 1
```
label_dict = {
    0: 'belt',
    1: 'no_belt',
    2: 'shoes',
    3: 'no_shoes',
    4: 'helmet',
    5: 'no_helmet',
    6: 'person'
}
```

### Label 2
```
label_dict = {
    0: 'belt',
    1: 'shoes',
    2: 'helmet',
    3: 'person'
}
```

### Dataset
AI 허브(공사 현장 안전장비 인식 이미지)

https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=163

roboflow(Personal Protective Equipment - Combined Model)

https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/browse?queryText=class%3A%22NO-Safety+Vest%22&pageSize=50&startingIndex=500&browseQuery=true

roboflow(Hard Hat Workers Dataset) - 안전모 미착용

https://public.roboflow.com/object-detection/hard-hat-workers/2


roboflow(clothes detect) - 안전조끼 미착용

https://universe.roboflow.com/zhang-ya-ying/clothes-detect-fevqm/dataset/5

roboflow(site2)

https://app.roboflow.com/changwoo-kim-vvfty/site2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

roboflow(whole_dataset) - 안전화 미착용

https://universe.roboflow.com/business-qcddc/whole_dataset/dataset/4


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

## 4. GUI (PyQt5)

```
pip install pyqt5
pip install pyside2
```
### 동영상 플레이어 참고 https://oceancoding.blogspot.com/2020/07/blog-post_22.html

<hr>

## 5. Reference

https://koreascience.kr/article/JAKO201915555313326.pdf
https://ysyblog.tistory.com/m/294 (Python email 이미지 첨부)



+ 이미지 트랙킹 
2023년 2월 23 ~ 2023년 2월 24일

시행착오
- detect_track.py를 보고 도저히 이해가 안되고 output을 가져다 쓰려고 했더니 12.mp4에서 출력이 되지 않음
- 그래서 자꾸 frame 이 어디서 저장되고 어떻게 처리되는지 찾으려고 함
- 못 찾아서 deep sort 공부를 더 함 (알고리즘, 클래스 다이어그램, 배경지식 등등)
- 공부를 하고 다시 코드를 보며 수정하고 기존 GUI와 합치는 과정에서 mp4파일에서 초반부엔 사람이 안나온다는게 문득 머릿속을 스쳐지나감
- mp4 파일을 detect_track.py에 넣고 output이 출력될 때 까지 기다림
- 기다렸더니 출력됨
- 이런 뻘짓을 했고 처음부터 초반부엔 output이 안나온다는걸 알았다면 아마 결과물은 나왔을거 같음
- 그래도 뻘짓을 했기에 deep sort 공부를 하게됨
- 뻘짓을 안했다면 결과물은 있었겠지만 머릿속에 남은게 없었을듯 이란 아쉬운 자기합리화를 하는 오전이었습니다
