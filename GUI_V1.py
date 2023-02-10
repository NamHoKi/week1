import sys
from PyQt5.QtWidgets import *

import torch.cuda
import os
import glob
import cv2
import numpy as np

# cvt_label_dict = {v: k for k, v in label_dict.items()}




class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # device setting
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model call
        self.model = None

        self.setupUI()


        self.label_dict = {
                0: 'belt',
                1: 'no_belt',
                2: 'hoes',
                3: 'no_shoes',
                4: 'helmet',
                5: 'no_helmet',
                6: 'person'
            }


    def setupUI(self):
        self.setGeometry(600, 400, 500, 500)
        btn1 = QPushButton("Model", self)
        btn1.move(100, 100)
        btn1.clicked.connect(self.set_model)

        btn2 = QPushButton("Image", self)
        btn2.move(100, 200)
        btn2.clicked.connect(self.show_image)

        btn3 = QPushButton("MP4", self)
        btn3.move(100, 300)
        btn3.clicked.connect(self.play_mp4)


    def set_model(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0] != '' and fname[0][-3:] == '.pt' :
            QMessageBox.information(self, 'Info', 'wait for setting model...')
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=fname[0])
            self.model.conf = 0.5  # NMS confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.to(self.DEVICE)
            QMessageBox.information(self, 'Info', 'Model setting complete')
        else :
            QMessageBox.information(self, 'Warning', 'Not Found .pt File')


    def get_result_image(self, image):
        # model input
        # 모델에 이미지를 넣어준다.
        output = self.model(image, size=640)
        # print(output.print())
        bbox_info = output.xyxy[0]  # bounding box의 결과를 추출
        # for문을 들어가서 우리가 원하는 결과를 뽑는다.

        for bbox in bbox_info:
            # bbox에서 x1, y1, x2, y2, score, label_number의 결과를 가지고 온다.
            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())

            score = bbox[4].item()
            label_number = int(bbox[5].item())
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, self.label_dict[label_number], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                cv2.putText(image, str(round(score, 4)), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            except Exception as e:
                print(e)

        return image


    def show_image(self):
        if self.check_model() :
            fname = QFileDialog.getOpenFileName(self)
            if fname[0] != '' and (fname[0][-4:] == '.jpg' or fname[0][-4:] == '.png') :
                image = cv2.imread(fname[0])

                image = self.get_result_image(image)

                cv2.imshow("test", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else :
                QMessageBox.information(self, 'Warning', 'Not Found Image')


    def check_model(self):
        if self.model is None :
            QMessageBox.information(self, 'Warning', 'Not Found Model')
            return False
        return True


    def play_mp4(self):
        if self.check_model() :
            model = self.model
            label_dict = self.label_dict
            fname = QFileDialog.getOpenFileName(self)
            if fname[0] != '' and fname[0][-4:] == '.mp4':
                Vid = cv2.VideoCapture(fname[0])

                if Vid.isOpened():
                    fps = Vid.get(cv2.CAP_PROP_FPS)
                    f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                    f_height = Vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

                    while Vid.isOpened():
                        ret, image = Vid.read()
                        if ret:
                            # model input
                            # 모델에 이미지를 넣어준다.
                            output = model(image, size=640)
                            # print(output.print())
                            bbox_info = output.xyxy[0]  # bounding box의 결과를 추출
                            # for문을 들어가서 우리가 원하는 결과를 뽑는다.

                            for bbox in bbox_info:
                                # bbox에서 x1, y1, x2, y2, score, label_number의 결과를 가지고 온다.
                                x1 = int(bbox[0].item())
                                y1 = int(bbox[1].item())
                                x2 = int(bbox[2].item())
                                y2 = int(bbox[3].item())

                                score = bbox[4].item()
                                label_number = int(bbox[5].item())
                                try:
                                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(image, label_dict[label_number], (int(x1), int(y1 - 5)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 0, 255), 2)
                                    cv2.putText(image, str(round(score, 4)), (int(x1), int(y1 - 25)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 0, 255), 2)
                                except Exception as e:
                                    print(e)

                            re_frame = cv2.resize(image, (round(f_width), round(f_height)))
                            cv2.imshow('Car_Video', re_frame)
                            key = cv2.waitKey(10)

                            if key == ord('q'):
                                break
                        else:
                            break

            Vid.release()
            cv2.destroyAllWindows()


    def play_webcam(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
