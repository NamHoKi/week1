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
import asyncio


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        # device setting
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model call
        self.model = None

        self.danger_count = 0
        self.danger_start_time = 0
        self.danger_last_time = 0
        self.danger_check = False
        self.recipient = 'skaghrl0@naver.com'

        # email 보내는 조건 설정
        self.send_limit_time = 5
        self.reset_limit_time = 3

        self.setupUI()

        self.label_dict = {
                0: 'belt',
                1: 'no_belt',
                2: 'shoes',
                3: 'no_shoes',
                4: 'helmet',
                5: 'no_helmet',
                6: 'person'
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
        btn5.clicked.connect(self.set_email)

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

        person = []
        no_person = []

        for bbox in bbox_info:
            # bbox에서 x1, y1, x2, y2, score, label_number의 결과를 가지고 온다.
            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())

            score = bbox[4].item()
            label_number = int(bbox[5].item())

            if label_number == 6:
                person.append([x1, y1, x2, y2, score])
            else:
                no_person.append([x1, y1, x2, y2, label_number])

        result_list = self.safe_check(person, no_person)
        danger_temp_check = False
        for r in result_list:
            x1, y1, x2, y2, score, result = r[0], r[1], r[2], r[3], r[4], r[5]

            if r[5] == 'Danger' :
                danger_temp_check = True

            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(image, r[5], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
                cv2.putText(image, str(round(r[4], 4)), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
            except Exception as e:
                print(e)

        return image, danger_temp_check


    def safe_check(self, person, no_person):
        result_list = []
        for p in person:
            result = '-'
            # print('='*10, p[0], p[1], p[2], p[3])
            for np in no_person:
                np_x, np_y = (np[0] + np[2]) / 2, (np[1] + np[3]) / 2
                # print('-' * 10, np[0], np[1], np[2], np[3], np_x, np_y)

                if p[0] <= np_x and p[2] >= np_x and p[1] <= np_y and p[3] >= np_y:
                    if np[4] == 1 or np[4] == 3 or np[4] == 5 :
                        result = "Danger"
            if result == '-' :
                result = "Safety"


            result_list.append([p[0], p[1], p[2], p[3], p[4], result])

        return result_list


    def show_image(self):
        if self.check_model() :
            fname = QFileDialog.getOpenFileName(self)
            if fname[0] != '' and (fname[0][-4:] == '.jpg' or fname[0][-4:] == '.png') :
                image = cv2.imread(fname[0])
                self.add_gui_console('IMAGE READ - ' + fname[0])

                image = self.get_result_image(image)[0]
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


    def play_mp4(self) :
        if self.check_model() :
            fname = QFileDialog.getOpenFileName(self)

            if fname[0] != '' and fname[0][-4:] == '.mp4':
                Vid = cv2.VideoCapture(fname[0])
                self.add_gui_console('MP4 READ - ' + fname[0])

                if Vid.isOpened():
                    while Vid.isOpened():
                        ret, frame = Vid.read()

                        if ret:
                            # model input
                            # 모델에 이미지를 넣어준다.
                            frame, danger_temp_check = self.get_result_image(frame)

                            if danger_temp_check:
                                self.danger_count += 1
                                if not self.danger_check :
                                    self.danger_check = True
                                    self.danger_start_time = time.time()
                                self.danger_last_time = time.time()

                                if self.danger_last_time - self.danger_start_time >= self.send_limit_time:
                                    cv2.imwrite('./image/email_image.png', frame)
                                    self.send_email()
                                    self.reset_danger()
                            else:
                                if time.time() - self.danger_last_time >= self.reset_limit_time:
                                    self.reset_danger()

                            frame = cv2.resize(frame, (1120, 630))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임에 색입히기
                            self.convertToQtFormat = QImage(frame.data, frame.shape[1],
                                                            frame.shape[0],
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
            self.reset_danger()


    def set_email(self):
        text = self.line_edit1.text()  # line_edit text 값 가져오기
        self.recipient = text
        self.text_label1.setText(self.recipient)  # label에 text 설정하기


    def add_gui_console(self, text) :
        origin_text = self.scroll_label.text()  # line_edit text 값 가져오기
        text = origin_text + '\n' + 'time - ' + str(time.time() - self.start_time) + '; GUI console : ' + text
        self.scroll_label.setText(text)  # label에 text 설정하기


    def send_email(self) :

        # 1. 이메일 전송 함수 정의
        def send_email(subject, message, recipient):
            # SMTP 서버 정보
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            smtp_user = "dbtmd324@gmail.com"
            smtp_pass = "your pass"

            # 이메일 보내기
            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.ehlo()
                server.starttls()
                server.login(smtp_user, smtp_pass)
                message = f"Subject: {subject}\n\n{message}"
                server.sendmail(smtp_user, recipient, message)
                print("Still not detected")
            except Exception as e:
                print(f"Failed to send email. Error: {e}")


        # 2. 알람 서비스 실행
        subject = "[Alert] Safety equipment not detected"
        message = "Not wearing safety gear is detected at the site."
        recipient = self.recipient
        send_email(subject, message, recipient)
        print("Alert email sent successfully.")
        self.add_gui_console('Send Email')


    def reset_danger(self):
        self.danger_count = 0
        self.danger_start_time = 0
        self.danger_last_time = 0
        self.danger_check = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
