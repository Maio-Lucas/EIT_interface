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


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # set the window title
        self.setWindowTitle('Hello World')
        
        # show the window
        self.show()
    
    def execute_window(self):
        app = QApplication(sys.argv)

        # create the main window
        window = MainWindow()

        # start the event loop
        sys.exit(app.exec())