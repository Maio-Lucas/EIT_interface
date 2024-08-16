import pyeit_controller
import pyqt_interface
import numpy as np

dados = np.loadtxt('interface_v01\dados_gravados_0123678c.txt')
(nframes,nmed) = dados.shape

"""Criar Selecionador de shape!!!"""