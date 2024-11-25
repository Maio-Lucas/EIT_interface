import pyeit_controller
import pyqt_interface
import numpy as np
from PyQt6.QtWidgets import QApplication
from pyqt_interface import MainWindow
import sys

#Start of all code by reading the txt file with the data
dados = np.loadtxt('dados_gravados_0123678c.txt')
(nframes,nmed) = dados.shape

#Instantiate PyQt application
app = QApplication(sys.argv)

#Call the developed window 
window = MainWindow(dados, nframes, method='bp')
window.show()

app.exec()

"""Criar Selecionador de shape!!!"""