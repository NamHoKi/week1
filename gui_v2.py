import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage

import smtplib
import torch.cuda
import cv2
import numpy as np
import time


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        # device setting
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model call
        self.model = None

        self.setupUI()

        # self.label_dict = {
        #         0: 'belt',
        #         1: 'no_belt',
        #         2: 'hoes',
        #         3: 'no_shoes',
        #         4: 'helmet',
        #         5: 'no_helmet',
        #         6: 'person'
        #     }

        self.label_dict = {
            0: 'belt',
            1: 'shoes',
            2: 'helmet',
            3: 'person'
        }


    def setupUI(self):
        # Set main window
        self.setWindowTitle("MS AI School")
        self.setWindowIcon(QIcon('image/icon.png'))
        self.setGeometry(100, 100, 1300, 800)
        self.setStyleSheet("color: white;"
                        "background-color: #333333")

        # Button 1
        btn1 = QPushButton("MODEL\nload", self)
        btn1.setStyleSheet("color: white;"
                        "background-color: #444444")
        btn1.move(10, 10)
        btn1.resize(130, 50)
        btn1.clicked.connect(self.set_model)
        # Button 2
        btn2 = QPushButton("IMAGE\nload", self)
        btn2.setStyleSheet("color: white;"
                           "background-color: #444444")
        btn2.move(10, 80)
        btn2.resize(130, 50)
        btn2.clicked.connect(self.show_image)
        # Button 3
        btn3 = QPushButton("MP4\nload", self)
        btn3.setStyleSheet("color: white;"
                           "background-color: #444444")
        btn3.move(10, 150)
        btn3.resize(130, 50)
        btn3.clicked.connect(self.play_mp4)
        self.mp4_stop = False
        # Button 4
        btn4 = QPushButton("MP4\nstop", self)
        btn4.setStyleSheet("color: white;"
                           "background-color: #444444")
        btn4.move(10, 220)
        btn4.resize(130, 50)
        btn4.clicked.connect(self.MP4_Stop)

        # Line edit
        self.line_edit1 = QLineEdit(self)
        self.line_edit1.move(10, 370)
        self.line_edit1.resize(130, 50)
        # Text label
        self.text_label1 = QLabel(self)
        self.text_label1.move(10, 440)
        self.text_label1.setText('Your E-mail')
        self.text_label1.resize(130, 50)
        # Button 5
        btn5 = QPushButton(self)
        btn5.resize(130, 50)
        btn5.setStyleSheet("color: white;"
                           "background-color: #444444")
        btn5.move(10, 510)
        btn5.setText('OK')
        btn5.clicked.connect(self.button_event)

        # Image window
        self.pixmap = QPixmap('image/black.png')
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.pixmap)  # 이미지 세팅
        self.image_label.setContentsMargins(10, 10, 10, 10)  # 1120 630
        self.image_label.resize(self.pixmap.width(), self.pixmap.height())
        self.image_label.move(140, 0)

        # Status Bar
        self.sb = self.statusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage('None')

        # Scroll Bar
        self.scrollArea1 = QScrollArea(self)
        self.scroll_label = QLabel('time - ' + str(time.time() - self.start_time) + '; GUI console : ' + 'Start MS AI School Team 1 Project')
        self.scroll_label.setStyleSheet("color: black;"
                           "background-color: #ffffff")
        self.scroll_label.resize(1200, 20000)
        self.scroll_label.setAlignment(Qt.AlignTop)
        self.scrollArea1.setWidget(self.scroll_label)
        self.scrollArea1.move(10, 650)
        self.scrollArea1.resize(1260, 130)
        self.scrollArea1.setStyleSheet("color: white;"
                           "background-color: #ffffff")

        self.show()


    # def set_tab1(self) :
    #     # Text : "Upload"
    #     label1 = QLabel(self)
    #     label1.setText('Upload')
    #     label1.move(40, 30)
    #
    #     # Button 1
    #     btn1 = QPushButton("Model", self)
    #     btn1.move(10, 50)
    #     btn1.clicked.connect(self.set_model)
    #     # Button 2
    #     btn2 = QPushButton("Image", self)
    #     btn2.move(10, 90)
    #     btn2.clicked.connect(self.show_image)
    #     # Button 3
    #     btn3 = QPushButton("MP4", self)
    #     btn3.move(10, 130)
    #     btn3.clicked.connect(self.play_mp4)
    #
    #     vbox = QVBoxLayout()
    #     vbox.addWidget(btn1)
    #     vbox.addWidget(btn2)
    #     vbox.addWidget(btn3)
    #     vbox.addWidget(label1)
    #
    #     # 위젯에 레이아웃 추가하기
    #     tab = QWidget()
    #     tab.setLayout(vbox)
    #     return tab
    #
    #
    # def set_tab2(self) :
    #     btn1 = QPushButton("Test", self)
    #     vbox = QVBoxLayout()
    #     vbox.addWidget(btn1)
    #
    #     # 위젯에 레이아웃 추가하기
    #     tab = QWidget()
    #     tab.setLayout(vbox)
    #     return tab


    def set_model(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0] != '' and fname[0][-3:] == '.pt' :
            QMessageBox.information(self, 'Info', 'Wait for Model setting')
            self.add_gui_console('UPLOADING : Wait for uploading model ...')
            self.sb.showMessage('UPLOADING : Wait for uploading model ...')
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=fname[0])
            self.model.conf = 0.5  # NMS confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.to(self.DEVICE)
            self.sb.showMessage('COMPLETE  --- Model setting')
            self.add_gui_console('COMPLETE  : Model setting')
            QMessageBox.information(self, 'Info', 'COMPLETE  : Model setting')
        else :
            self.sb.showMessage('WARNING   --- Not Found (*.pt) File')
            self.add_gui_console('WARNING   --- Not Found (*.pt) File')
            QMessageBox.information(self, 'Warning', 'WARNING   : Not Found (*.pt) File')


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
                self.add_gui_console(str(e))

        return image


    def show_image(self):
        if self.check_model() :
            fname = QFileDialog.getOpenFileName(self)
            if fname[0] != '' and (fname[0][-4:] == '.jpg' or fname[0][-4:] == '.png') :
                image = cv2.imread(fname[0])
                self.add_gui_console('IMAGE READ - ' + fname[0])

                # cv2.imwrite('image/test.jpg', cv2.resize(self.get_result_image(image), 1120, 630))
                # self.pixmap = QPixmap('image/test.jpg')

                image = self.get_result_image(image)
                image = cv2.resize(image, (1120, 630))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 프레임에 색입히기
                self.convertToQtFormat = QImage(image.data, image.shape[1],
                                                image.shape[0],
                                                QImage.Format_RGB888)
                self.pixmap = QPixmap(self.convertToQtFormat)
                self.image_label.setPixmap(self.pixmap)  # 이미지 세팅
                self.image_label.setContentsMargins(10, 10, 10, 10)  # 여백 설정
                self.image_label.resize(self.pixmap.width(), self.pixmap.height())
                self.image_label.move(140, 0)
                self.sb.showMessage('COMPLETE  --- Show image')
                self.add_gui_console('COMPLETE  --- Show image')
            else :
                self.add_gui_console('WARNING   --- Not Found (Image) File')
                self.sb.showMessage('WARNING   --- Not Found (Image) File')
                QMessageBox.information(self, 'Warning', 'WARNING   : Not Found (Image) File')


    def check_model(self):
        if self.model is None :
            self.add_gui_console('WARNING   --- Not Found Model')
            QMessageBox.information(self, 'Warning', 'Not Found Model')
            return False
        return True


    def play_mp4(self):
        if self.check_model() :
            fname = QFileDialog.getOpenFileName(self)

            if fname[0] != '' and fname[0][-4:] == '.mp4':
                Vid = cv2.VideoCapture(fname[0])
                self.add_gui_console('MP4 READ - ' + fname[0])

                if Vid.isOpened():
                    while Vid.isOpened():
                        ret, image = Vid.read()

                        if ret:
                            # model input
                            # 모델에 이미지를 넣어준다.
                            image = self.get_result_image(image)
                            image = cv2.resize(image, (1120, 630))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 프레임에 색입히기
                            self.convertToQtFormat = QImage(image.data, image.shape[1],
                                                            image.shape[0],
                                                            QImage.Format_RGB888)
                            self.pixmap = QPixmap(self.convertToQtFormat)
                            self.image_label.setPixmap(self.pixmap)  # 이미지 세팅
                            self.image_label.setContentsMargins(10, 10, 10, 10)  # 여백 설정
                            self.image_label.resize(self.pixmap.width(), self.pixmap.height())
                            self.image_label.move(140, 0)

                            cv2.waitKey(5)

                            if self.mp4_stop :
                                print('KeyboardInterrupt : "mp4 stop"')
                                self.add_gui_console('KeyboardInterrupt : MP4 STOP')
                                break

                        else:
                            break

                    Vid.release()
                    cv2.destroyAllWindows()
                    self.mp4_stop = False


    def MP4_Stop(self) :
        if self.mp4_stop == False :
            self.mp4_stop = True


    def button_event(self):
        text = self.line_edit1.text()  # line_edit text 값 가져오기
        # self.add_gui_console('KeyboardInterrupt : MP4 STOP')
        self.text_label1.setText(text)  # label에 text 설정하기


    def add_gui_console(self, text) :
        origin_text = self.scroll_label.text()  # line_edit text 값 가져오기
        text = origin_text + '\n' + 'time - ' + str(time.time() - self.start_time) + '; GUI console : ' + text
        self.scroll_label.setText(text)  # label에 text 설정하기


    # def alaram(self) :
    #     def send_email(subject, message, recipient):
    #         # SMTP 서버 정보
    #         smtp_server = "smtp.gmail.com"
    #         smtp_port = 587
    #         smtp_user = "skaghrl0@gmail.com"
    #         smtp_pass = "gnoukvtchqfvyvpm"
    #
    #         # 이메일 보내기
    #         try:
    #             server = smtplib.SMTP(smtp_server, smtp_port)
    #             server.ehlo()
    #             server.starttls()
    #             server.login(smtp_user, smtp_pass)
    #             message = f"Subject: {subject}\n\n{message}"
    #             server.sendmail(smtp_user, recipient, message)
    #             print("Still not detected")
    #         except Exception as e:
    #             print(f"Failed to send email. Error: {e}")
    #
    #     def check_safety_equipment():
    #         # 안전 장구류 미착용 확인 코드
    #         return False
    #
    #     while True:
    #         safety_equipment_detected = check_safety_equipment()
    #         if not safety_equipment_detected == True:
    #             print('Safety equipment not detected. Checking...')
    #             time.sleep(30)  # 1 minutes
    #             if not safety_equipment_detected == True:
    #                 subject = "[Alert] Safety equipment not detected"
    #                 message = "Not wearing safety gear is detected at the site."
    #                 recipient = "skaghrl0@naver.com"
    #                 send_email(subject, message, recipient)
    #                 print("Alert email sent successfully.")
    #             else:
    #                 print('Now detected!')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
