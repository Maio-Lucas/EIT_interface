import sys
import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import (
    QPixmap,
    QPalette,
    QColor,
    QFont,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyeit_controller import EITsolver

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=2.5, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

class MainWindow(QMainWindow):

    def __init__(self, data, nframes):
        super(MainWindow, self).__init__()

        # MainWindow configuration
        self.setWindowTitle("EITduino")
        self.setMinimumSize(QSize(800, 600))
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('white'))
        self.setPalette(palette)

        # Application Header layout
        layout_header = QGridLayout()
        layout_header.setColumnStretch(0,1) # logo
        layout_header.setColumnStretch(1,10) # header text

        logoUFABC = QLabel("")
        logoUFABC.setPixmap(QPixmap('logo_UFABC.png').scaled(75,75))
        headerText = QLabel("EITduino")
        headerText.setStyleSheet("QLabel { color : #006633; font size : 40; }"); # UFABC standard color
        fHeader = QFont("Humanst777", 50, weight=625) # UFABC standard font
        headerText.setFont(fHeader)

        # layout_header.addWidget(logoUFABC,0,0)
        layout_header.addWidget(headerText,0,1)

        # User interface layout
        layout_gui = QGridLayout()
        layout_gui.setColumnStretch(0,1) # controls tabs
        layout_gui.setColumnMinimumWidth(0,400)
        layout_gui.setColumnStretch(1,10) # image

        self.eitMeasurementsSE = MplCanvas(self, width=10, height=2) # plot
        self.eitMeasurementsDiff = MplCanvas(self, width=10, height=2) # plot

        layout_measurements = QVBoxLayout()
        layout_measurements.addWidget(self.eitMeasurementsSE)
        layout_measurements.addWidget(self.eitMeasurementsDiff)
        measurements_widget = QWidget()
        measurements_widget.setLayout(layout_measurements)

        tab1 = QWidget()
        tab1_layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(Color('red'), 'Solver Config')
        tabs.addTab(measurements_widget, 'Measurements')
        tabs.addTab(Color('gray'), 'Controls')

        buttonBP = QPushButton('BP', tabs)
        buttonBP.clicked.connect(on_button_click)
        buttonBP.setGeometry(0, 0, 100, 50)
        self.eitImage = MplCanvas(self) # plot

        layout_gui.addWidget(tabs)
        layout_gui.addWidget(self.eitImage,0,1)

        # Main window layout
        layout_main = QGridLayout()
        layout_main.setRowStretch(0,1) # header layout
        layout_main.setRowStretch(1,10) # gui layout
        layout_main.addLayout(layout_header,0,0)
        layout_main.addLayout(layout_gui,1,0)
        
        # Defining central widget
        main_widget = QWidget()
        main_widget.setLayout(layout_main)
        self.setCentralWidget(main_widget)

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(lambda: self.update_plot(data, nframes))
        self.timer.start()

        # Other commands
        self.mySover = EITsolver(method='greit', h0=0.1)
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0
        self.init_plots()
        self.update_plot(data, nframes)
    
    def init_plots(self):
        self._plotImage_ref = self.eitImage.axes.imshow(np.zeros((32,32)), vmin=-0.75, vmax=0.75, origin='lower')
        self.eitImage.fig.colorbar(self._plotImage_ref)
    
    def update_plot(self, data, nframes):
        self.dataSE = data[self.frameCounter]
        if self.frameCounter==0:
            self.mySover.setVref(self.dataSE)
        
        self.dataDiff = self.mySover.se_to_diff(self.dataSE)

        if self._plotSE_ref is None: # 1st time, new plot
            self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE)[0]
        else: # update data
            self._plotSE_ref.set_ydata(self.dataSE)

        if self._plotDiff_ref is None: # 1st time, new plot
            self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(self.dataDiff)[0]
        else: # update data
            self._plotDiff_ref.set_ydata(self.dataDiff)
        
        self.mySover.updateImage(self.dataSE, self._plotImage_ref)
        
        # Update titles
        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")
        self.eitMeasurementsSE.axes.set_title(f"Single Ended Measurements ({self.frameCounter})")
        # Trigger plots to update and redraw.
        self.eitMeasurementsSE.axes.set_ylim([self.dataSE.min(), self.dataSE.max()])
        self.eitMeasurementsDiff.axes.set_ylim([self.dataDiff.min(), self.dataDiff.max()])
        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()
        self.frameCounter = self.frameCounter + 1
        if self.frameCounter==nframes:
            self.frameCounter=0

    def update_plot_random(self, nframes):
        self.data = np.random.random_sample((32,32)) # generate random imagem for debug
        self.dataSE = np.random.random_sample((64,1)) # generate random measurements for debug
        self.dataDiff = np.random.random_sample((64,1)) # generate random measurements for debug

        if self._plotImage_ref is None: # 1st time, new plot
            self._plotImage_ref = self.eitImage.axes.imshow(self.data)
        else: # update data
            self._plotImage_ref.set_data(self.data)

        if self._plotSE_ref is None: # 1st time, new plot
            self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE)[0]
        else: # update data
            self._plotSE_ref.set_ydata(self.dataSE)

        if self._plotDiff_ref is None: # 1st time, new plot
            self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(self.dataDiff)[0]
        else: # update data
            self._plotDiff_ref.set_ydata(self.dataDiff)
        
        # Update titles
        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")
        self.eitMeasurementsSE.axes.set_title(f"Single Ended Measurements ({self.frameCounter})")
        self.eitImage.fig.colorbar(self._plotImage_ref)
        # Trigger plots to update and redraw.
        self.eitImage.draw()
        self.eitImage.fig.colorbar(self._plotImage_ref)
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()
        self.frameCounter = self.frameCounter + 1
        if self.frameCounter==nframes:
            self.frameCounter=0

def on_button_click():
    print("Button clicked!")
