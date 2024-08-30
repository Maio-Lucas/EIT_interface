import pyeit_controller
import pyqt_interface
import numpy as np
from PyQt6.QtWidgets import QApplication
from pyqt_interface import MainWindow
import sys

dados = np.loadtxt('dados_gravados_0123678c.txt')
(nframes,nmed) = dados.shape


app = QApplication(sys.argv)

window = MainWindow(dados, nframes)
window.show()

app.exec()

"""Criar Selecionador de shape!!!"""