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

def on_button_click():
    print("Button clicked!")

app = QApplication(sys.argv)

# Create the main window
window = QMainWindow()
window.setWindowTitle('PyEIT Interface')
window.setGeometry(0, 0, 300, 300)

# Create a button widget
buttonBP = QPushButton('Click Me', window)
buttonBP.clicked.connect(on_button_click)
buttonBP.setGeometry(100, 100, 100, 50)

# Show the window
window.show()

# Run the application event loop
app.exec()