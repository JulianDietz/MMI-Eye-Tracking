# -*- coding: utf-8 -*-
import json
import sys
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton)
import pyqtgraph as pg
import numpy as np


class CaputeClient(QWidget):
    def __init__(self, cam_dict, classes, change_activity_calback, start_logging_calback, stop_logging_calback,
                 filename_callback):
        super().__init__()
        self.cam_dict = cam_dict
        self.classes = classes
        self.prediction_plots = {}
        self.load_and_show_ui()
        self.change_activity = change_activity_calback
        self.create_activity_buttons(classes)
        self.start_logging_calback = start_logging_calback
        self.stop_logging_calback = stop_logging_calback
        self.filename_callback = filename_callback
        self.win.selDirButton.clicked.connect(self.selectFilePath)
        self.csv_buttons()

    def csv_buttons(self):
        self.win.recordButton.clicked.connect(self.record_csv)

    def record_csv(self):
        if self.win.recordButton.text() == 'Start recording':
            participant = self.win.probandLineEdit.text()
            self.startRecordTimer()
            self.win.recordButton.setText('Stop recording')
            filepath = self.filename_callback(self.record_dir, participant)
            self.start_logging_calback()
            self.win.filenameLineEdit.setText(filepath)

        elif self.win.recordButton.text() == 'Stop recording':
            self.stopTimer()
            self.win.recordButton.setText('Start recording')
            self.stop_logging_calback()

    def startRecordTimer(self):
        self.record_timer = QTimer(self)
        self.timer = self.win.lcdTimer
        self.record_timer.start(100)
        self.record_timer.timeout.connect(self.updateTimer)

    def updateTimer(self):
        self.timer.display(self.timer.value() + 0.1)

    def stopTimer(self):
        timer = self.win.lcdTimer()
        timer.display(0.0)
        self.record_timer.stop()

    # loads and shows the template file
    def load_and_show_ui(self):
        self.win = uic.loadUi('CaptureTool.ui')
        self.win.closeEvent = self.closeEvent
        self.kitchen_widget()
        self.plot_widget()
        self.win.show()

    def closeEvent(self, event):
        pass

    def plot_widget(self):
        my_plot = pg.PlotWidget()
        layout = self.win.gridLayout_2
        layout.addWidget(my_plot)
        x = np.arange(6)
        y = [0, 0, 0, 0, 0, 0]

        self.prediction_plots['cam_1'] = pg.BarGraphItem(x=x + 0.1, height=y, width=0.2, brush='r', name='cam_1')
        my_plot.addItem(self.prediction_plots['cam_1'])

        self.prediction_plots['cam_2'] = pg.BarGraphItem(x=x + 0.4, height=y, width=0.2, brush='g', name='cam_2')
        my_plot.addItem(self.prediction_plots['cam_2'])

        self.prediction_plots['cam_3'] = pg.BarGraphItem(x=x + 0.7, height=y, width=0.2, brush='b', name='cam_3')
        my_plot.addItem(self.prediction_plots['cam_3'])

        my_plot.setTitle('Activity Prediction')
        my_plot.setLabel('right', "Percentage")

        text = ""
        for i in range(len(self.classes)):
            text += str(i) + '-' + str(i + 1) + ': ' + self.classes[i] + ',  '

        my_plot.setLabel('bottom', text)
        my_plot.setXRange(-0.1, 6.1, padding=0)
        my_plot.setYRange(0, 100, padding=0)

    def updatePlot(self, prediction):
        for cam in prediction:
            self.prediction_plots[cam].setOpts(height=list(prediction[cam].values()))
            # self.prediction_plots[cam].update()

    def create_activity_buttons(self, classes):
        self.current_activity = self.win.currentActivityLineEdit
        layout = self.win.gridLayout_class
        for activity in classes:
            button = QPushButton(activity)
            button.clicked.connect(self.activity_changed)
            layout.addWidget(button)

    def activity_changed(self):
        activity_class = self.sender().text()
        self.current_activity.setText(activity_class)
        self.change_activity(activity_class)

    def kitchen_widget(self):
        lay = self.win.scrollArea
        self.kitchen = DrawKitchenWidget()
        lay.setWidget(self.kitchen)

    def setAOI(self, left, right):
        self.kitchen.setAOI(left, right)

    def selectFilePath(self):
        self.record_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))


class DrawKitchenWidget(QWidget):
    height = 180
    width = 1000
    scale = 4
    config = {}
    x_offset = 280
    textHitRight = ""
    textHitLeft = ""

    def __init__(self, *args, **kwargs):
        super(DrawKitchenWidget, self).__init__(*args, **kwargs)
        self.loadConfig()
        self.setFixedSize(self.width, self.height)

    def loadConfig(self):
        with open('config/aoi_config_filled.json') as json_file:
            data = json.load(json_file)
            for item in data:
                self.config[item['name']] = item

    def drawConfig(self, painter):
        for key in self.config:
            item = self.config[key]
            col = QColor(255, 0, 0)
            painter.setPen(col)
            if 'hit' in item and item['hit']:
                painter.setBrush(QtCore.Qt.blue)
            else:
                painter.setBrush(Qt.NoBrush)
            if item['aoi_y_2'] == item['aoi_y_1']:
                # draws top Arbeitsfläche view
                if key not in ['Schrankunterseiten', 'Boden', 'Decke', 'Arbeitsfläche_1_Vorne',
                               'Arbeitsfläche_Spüle_Vorne', 'Arbeitsfläche_2_Vorne', 'Arbeitsfläche_Ofen_Vorne',
                               'Arbeitsfläche_3_Vorne']:
                    hoehe = (item['aoi_z_2'] - item['aoi_z_0']) / self.scale
                    ystart = item['aoi_z_0']
                    breite = (item['aoi_x_1'] - item['aoi_x_0']) / self.scale
                    x = item['aoi_x_0'] / self.scale + self.x_offset
                    y = self.height - (ystart / self.scale) - hoehe - 40
                else:
                    continue
            else:
                # draws front Küchen view
                if key not in ['Mauer_Links', 'Mauer_rechts', 'Hinten']:
                    breite = (item['aoi_x_1'] - item['aoi_x_0']) / self.scale
                    hoehe = (item['aoi_y_2'] - item['aoi_y_1']) / self.scale
                    x = item['aoi_x_0'] / self.scale
                    y = self.height - (item['aoi_y_0'] / self.scale) - hoehe
                else:
                    continue
            painter.drawRect(x, y, breite, hoehe)

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawConfig(painter)

        painter.setPen(QColor(0, 255, 0))
        painter.setFont(QFont('Decorative', 12))
        painter.drawText(QRect(self.x_offset * 2, 100, 200, self.height), Qt.AlignTop, self.textHitLeft)
        painter.drawText(QRect(self.x_offset * 3, 100, 200, self.height), Qt.AlignTop, self.textHitRight)
        painter.end()

    def setAOI(self, left, right):
        self.textHitRight = '\n'.join(right)
        self.textHitLeft = '\n'.join(left)
        for item in self.config:
            self.config[item]['hit'] = False
        for item in left:
            self.config[item]['hit'] = True
        for item in right:
            self.config[item]['hit'] = True
        self.update()


def setupCaptureTool(cam_dict, activity_classes, change_activity, start_logging, end_logging, filename_callback):
    global client
    app = QApplication(sys.argv)
    client = CaputeClient(cam_dict, activity_classes, change_activity, start_logging, end_logging, filename_callback)

    sys.exit(app.exec_())


def setAOI(left, right):
    aoi_right = []
    aoi_left = []
    for aoi in left:
        aoi_left.append(aoi[0])
    for aoi in right:
        aoi_right.append(aoi[0])
    client.setAOI(aoi_left, aoi_right)


def setPrediction(prediction):
    client.updatePlot(prediction)


if __name__ == '__main__':
    setupCaptureTool()
