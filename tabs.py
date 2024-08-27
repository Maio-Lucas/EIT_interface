import sys
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
from PyQt6.QtGui import (
    QPixmap,
    QPalette,
    QColor,
    QFont,
)

# Create the main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt6 QTabWidget Example")
        self.setGeometry(100, 100, 400, 300)

        # Create a QTabWidget
        tab_widget = QTabWidget()

        # Create tab 1
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1.setLayout(tab1_layout)

        buttonBP = QPushButton('BP', tab1)
        buttonJAC = QPushButton('JAC', tab1)
        buttonGREIT = QPushButton('GREIT', tab1)

        buttonBP.clicked.connect(on_button_click)
        buttonBP.setGeometry(0, 0, 100, 50)

        buttonJAC.clicked.connect(on_button_click)
        buttonJAC.setGeometry(0, 60, 100, 50)

        buttonGREIT.clicked.connect(on_button_click)
        buttonGREIT.setGeometry(0, 120, 100, 50)

        # Create tab 2
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        tab2_layout.addWidget(QLabel("Content of Tab 2"))
        tab2.setLayout(tab2_layout)

        # Create tab 3
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3_layout.addWidget(QLabel("Content of Tab 3"))
        tab3.setLayout(tab3_layout)

        # Add tabs to the QTabWidget
        tab_widget.addTab(tab1, "Tab 1")
        tab_widget.addTab(tab2, "Tab 2")
        tab_widget.addTab(tab3, "Tab 3")

        # Set the QTabWidget as the central widget of the main window
        self.setCentralWidget(tab_widget)

def on_button_click():
    print("Button clicked!")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
