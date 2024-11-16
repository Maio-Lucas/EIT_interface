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
    QComboBox,
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
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from pyeit.eit.interp2d import sim2pts
from pyeit_controller import EITsolver

i = 0

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

    #Initiate the window containing the graphs data
    def __init__(self, data, nframes, method='greit'):
        self.data = data

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
        self.layout_gui = QGridLayout()
        self.layout_gui.setColumnStretch(0,1) # controls tabs
        self.layout_gui.setColumnMinimumWidth(0,400)
        self.layout_gui.setColumnStretch(1,10) # image

        self.eitMeasurementsSE = MplCanvas(self, width=10, height=2) # plot
        self.eitMeasurementsDiff = MplCanvas(self, width=10, height=2) # plot

        layout_measurements = QVBoxLayout()
        layout_measurements.addWidget(self.eitMeasurementsSE)
        layout_measurements.addWidget(self.eitMeasurementsDiff)
        measurements_widget = QWidget()
        measurements_widget.setLayout(layout_measurements)

        self.tabs = QTabWidget()

        # Create first tab content
        tabConfig = QWidget()
        tabConfig_layout = QVBoxLayout()
        tabConfig_label = Color('red')
        tabConfig_layout.addWidget(tabConfig_label)
        tabConfig.setLayout(tabConfig_layout)

        buttonBP = QPushButton('BP', tabConfig)
        buttonBP.clicked.connect(lambda: on_button_click(self, data, nframes, button='bp'))
        buttonBP.setGeometry(15, 25, 50, 25)

        buttonJAC = QPushButton('JAC', tabConfig)
        buttonJAC.clicked.connect(lambda: on_button_click(self, data, nframes, button='jac'))
        buttonJAC.setGeometry(15, 55, 50, 25)

        buttonGREIT = QPushButton('GREIT', tabConfig)
        buttonGREIT.clicked.connect(lambda: on_button_click(self, data, nframes, button='greit'))
        buttonGREIT.setGeometry(15, 85, 50, 25)

        shapeSelector = QComboBox(tabConfig)
        shapeSelector.addItems(["Circle", "Ellipse", "Rectangle"])
        shapeSelector.setGeometry(200, 25, 150, 25)

        shapeSelector.currentIndexChanged.connect( self.index_changed )

        # There is an alternate signal to send the text.
        shapeSelector.currentTextChanged.connect( self.text_changed )

        self.tabs.addTab(tabConfig, 'Solver Config')

        self.tabs.addTab(measurements_widget, 'Measurements')

        self.tabs.addTab(Color('gray'), 'Controls')

        self.eitImage = MplCanvas(self) # plot

        self.layout_gui.addWidget(self.tabs)
        self.layout_gui.addWidget(self.eitImage,0,1)

        # Main window layout
        layout_main = QGridLayout()
        layout_main.setRowStretch(0,1) # header layout
        layout_main.setRowStretch(1,10) # gui layout
        layout_main.addLayout(layout_header,0,0)
        layout_main.addLayout(self.layout_gui,1,0)
        
        # Defining central widget
        main_widget = QWidget()
        main_widget.setLayout(layout_main)
        self.setCentralWidget(main_widget)

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: self.update_plot(data, nframes, method=method))
        self.timer.start()
        
        # Other commands
        self.mySolver = EITsolver(method=method, h0=0.1)
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0
        self.init_plots(data=data, method=method)
        self.update_plot(data, nframes, method=method)

    def update_solver(self, data, nframes, method='greit'):
        """Reinitialize the solver with the new method."""
        self.eitMeasurementsSE = MplCanvas(self, width=10, height=2) # plot
        self.eitMeasurementsDiff = MplCanvas(self, width=10, height=2) # plot

        self.eitImage = MplCanvas(self)
        self.layout_gui.addWidget(self.eitImage,0,1)

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: self.update_plot(data, nframes, method=method))
        self.timer.start()

        self.mySolver = EITsolver(method=method, h0=0.1)
        print(f"Solver updated to use method: {method}")
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0
        self.init_plots(data=data, method=method)
        self.update_plot(data, nframes, method=method)
    
    def init_plots(self, data, method='greit'):
        
        self.dataSE = data[self.frameCounter]
        if self.frameCounter==0:
            self.mySolver.setVref(self.dataSE)
            print("aqui")

        if method=='greit': 
            self._plotImage_ref = self.eitImage.axes.imshow(np.zeros((32,32)), vmin=-0.75, vmax=0.75, origin='lower')
            self.eitImage.fig.colorbar(self._plotImage_ref)
        
        elif method == 'jac' or method == 'bp':

            self.mySolver.setframes(Vse=self.data[self.frameCounter], method=method)

            if method=='jac':

                pts = self.mySolver.mesh_obj.node
                tri = self.mySolver.mesh_obj.element

                ds_n = sim2pts(pts, tri, np.real(self.mySolver.ds_med_frame))
                # draw
                self._plotImage_ref = self.eitImage.axes.tripcolor(pts[:, 0], pts[:, 1], tri, ds_n, shading="flat")
                self.eitImage.fig.colorbar(self._plotImage_ref)

            else:
                fig = plt.figure(figsize=(6, 4.5))
                ax1 = plt.gca()
    
    def update_plot(self, data, nframes, method):
        
        self.dataSE = data[self.frameCounter]
        self.dataDiff = self.mySolver.se_to_diff(self.dataSE)

        if self._plotSE_ref is None: # 1st time, new plot
            self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE)[0]
        else: # update data
            self._plotSE_ref.set_ydata(self.dataSE)

        if self._plotDiff_ref is None: # 1st time, new plot
            self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(self.dataDiff)[0]

        else: # update data
            self._plotDiff_ref.set_ydata(self.dataDiff)

        self.mySolver.updateImage(self.dataSE, method , self._plotImage_ref)
        
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

    def index_changed(window_instance, data, nframes):
        global method
        method = method
        print(f"Method set to: {method}")
        window_instance.update_solver(data, nframes, method)

    def text_changed(window_instance, data, nframes):
        global method
        method = method
        print(f"Method set to: {method}")
        window_instance.update_solver(data, nframes, method)

def on_button_click(window_instance, data, nframes, button='greit'):
    global method
    method = button
    print(f"Method set to: {method}")
    window_instance.update_solver(data, nframes, method)



