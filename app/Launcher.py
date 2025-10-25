import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import numpy as np

# Load the data from the stored file, this will need to change for real time data.
dados = np.loadtxt('stored_data/dados_gravados_0123678c.txt')
(nframes, nmed) = dados.shape
print('chegamos aqui')
#Here is defined the Launch Window class, inheriting methods from QMainWindow.
class LaunchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print('agora aqui')
        #Set name and fixed size
        self.setWindowTitle("EIT Interface Launcher")
        self.setFixedSize(400, 400)

        #The normal order is to define the items that will be in the
