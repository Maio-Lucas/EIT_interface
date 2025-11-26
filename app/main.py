import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import numpy as np

# Load the same data
dados = np.loadtxt('stored_data/dados_gravados_0123678c.txt')
(nframes, nmed) = dados.shape

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EIT Interface Launcher")
        self.setFixedSize(300, 150)

        # Buttons
        btn_pyqt6 = QPushButton("Launch PyQt6 (Matplotlib)")
        btn_pyqtgraph = QPushButton("Launch PyQtGraph")

        btn_pyqt6.clicked.connect(self.launch_pyqt6)
        btn_pyqtgraph.clicked.connect(self.launch_pyqtgraph)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(btn_pyqt6)
        layout.addWidget(btn_pyqtgraph)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def launch_pyqt6(self):
        from pyqt_interface import MainWindow
        self.app = MainWindow(dados, nframes, method='bp')
        self.app.show()
        self.close()

    def launch_pyqtgraph(self):
        from pyqtgraph_interface import MainWindowPG 
        self.app = MainWindowPG(dados, nframes, method='bp')
        self.app.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec())
