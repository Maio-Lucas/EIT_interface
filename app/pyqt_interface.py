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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from pyeit.eit.interp2d import sim2pts
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

    #Initiate the window containing the graphs data
    def __init__(self, data, nframes, method='greit'):
        
        self.data = data
        self.nframes = nframes
        self.method = method
        self._colorbar_ref = None


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
        pix = QPixmap('images/logo_UFABC.png')
        if not pix.isNull():
            logoUFABC.setPixmap(pix.scaled(75, 75))
        else:
            print("Warning: logo_UFABC.png not found or invalid.")

        headerText = QLabel("EITduino")
        headerText.setStyleSheet("QLabel { color : #006633; font size : 40; }"); # UFABC standard color
        fHeader = QFont("Humanst777", 50, weight=625) # UFABC standard font
        headerText.setFont(fHeader)

        layout_header.addWidget(logoUFABC,0,0)
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
        buttonBP.clicked.connect(lambda: self.update_solver('bp'))
        buttonBP.setGeometry(15, 25, 50, 25)

        buttonJAC = QPushButton('JAC', tabConfig)
        buttonJAC.clicked.connect(lambda: self.update_solver('jac'))
        buttonJAC.setGeometry(15, 55, 50, 25)

        buttonGREIT = QPushButton('GREIT', tabConfig)
        buttonGREIT.clicked.connect(lambda: self.update_solver('greit'))
        buttonGREIT.setGeometry(15, 85, 50, 25)

        """ shapeSelector = QComboBox(tabConfig)
        shapeSelector.addItems(["Circle", "Ellipse", "Rectangle"])
        shapeSelector.setGeometry(200, 25, 150, 25)

        shapeSelector.currentIndexChanged.connect( self.index_changed )

        # There is an alternate signal to send the text.
        shapeSelector.currentTextChanged.connect( self.text_changed ) """

        self.tabs.addTab(tabConfig, 'Solver Config')

        self.tabs.addTab(measurements_widget, 'Measurements')

        self.tabs.addTab(Color('gray'), 'Controls')

        self.eitImage = MplCanvas(self) # plot

        # Referências para colorbar e eixo da colorbar
        self._colorbar_ref = None
        self._cbar_ax = None

        # Limites fixos do eixo de imagem
        self._ax_xlim = None
        self._ax_ylim = None

        # Evitar alterações automáticas de layout
        try:
            self.eitImage.fig.set_constrained_layout(False)
        except Exception:
            pass

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

        # # Setup a timer to trigger the redraw by calling update_plot.
        # self.timer = QTimer()
        # self.timer.setInterval(50)
        # self.timer.timeout.connect(lambda: self.update_plot(data, nframes, method=method))
        # self.timer.start()

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()
        
        # Other commands
        self.mySolver = EITsolver(method=method, h0=0.1)
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0
        self.init_plots(data=data, method=method)
        self.update_plot(data, nframes, method=method)

    def _on_timer(self):
        # Este método substitui o lambda
        # Ele usa os atributos atuais da classe
        self.update_plot(self.data, self.nframes, method=self.method)

    def _attach_or_update_colorbar(self, mappable):
        """
        Cria a colorbar uma vez (em um eixo fixo ao lado) e depois apenas atualiza.
        Evita recriar e mudar o layout do eixo principal.
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if self._colorbar_ref is None or self._cbar_ax is None:
            # cria um eixo de colorbar fixo à direita
            divider = make_axes_locatable(self.eitImage.axes)
            self._cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
            self._colorbar_ref = self.eitImage.fig.colorbar(mappable, cax=self._cbar_ax)
        else:
            # apenas reaponta a colorbar para o novo "mappable"
            self._colorbar_ref.update_normal(mappable)

    def _lock_axes_extent(self):
        """
        Define limites consistentes e aspecto igual para o eixo de imagem,
        em função da malha atual (pts).
        """
        pts = self.mySolver.mesh_obj.node
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        span = max(x_max - x_min, y_max - y_min)
        margin = 0.05 * span
        self._ax_xlim = (x_min - margin, x_max + margin)
        self._ax_ylim = (y_min - margin, y_max + margin)

        ax = self.eitImage.axes
        ax.set_xlim(*self._ax_xlim)
        ax.set_ylim(*self._ax_ylim)
        ax.set_aspect('equal', adjustable='box')  # círculo não vira elipse
    
    def update_solver(self, new_method: str):
        """Troca de método sem recriar timer/duplicar widgets."""
        if new_method == self.method:
            return

        # pare o timer atual
        self.timer.stop()

        # atualize o estado
        self.method = new_method
        print(f"Solver updated to use method: {self.method}")

        # limpe os eixos existentes (não recrie canvases!)
        self.eitImage.axes.clear()
        self.eitMeasurementsSE.axes.clear()
        self.eitMeasurementsDiff.axes.clear()

        
        # Remover colorbar antiga (se existia)
        if self._colorbar_ref is not None:
            try:
                self._colorbar_ref.remove()
            except Exception:
                pass
            self._colorbar_ref = None
            self._cbar_ax = None


        # recrie o solver com o novo método (reaproveitando UI)
        self.mySolver.recreate_mesh(method=self.method, h0=0.1)

        # zere referências e contador
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0

        # re‑inicialize os plots e redesenhe
        self.init_plots(data=self.data, method=self.method)
        self.update_plot(self.data, self.nframes, method=self.method)

        # retome o timer
        self.timer.start()

    
    def _attach_colorbar(self, mappable):
        # Remove colorbar anterior, se existir
        if self._colorbar_ref is not None:
            try:
                self._colorbar_ref.remove()
            except Exception:
                pass
            self._colorbar_ref = None
        # Cria nova colorbar e guarda referência
        self._colorbar_ref = self.eitImage.fig.colorbar(mappable, ax=self.eitImage.axes)


    def init_plots(self, data, method='greit'):
        # Frame de referência e Vref
        self.dataSE = data[self.frameCounter]
        if self.frameCounter == 0:
            self.mySolver.setVref(self.dataSE)

        # --------- Curvas de medidas (SE e Diff) ---------
        # Cria as curvas na primeira vez
        self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE, lw=1)[0]
        diff0 = self.mySolver.se_to_diff(self.dataSE)
        self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(diff0, lw=1)[0]

        # Define limites estáveis (uma vez)
        try:
            se_min, se_max = float(self.data.min()), float(self.data.max())
        except Exception:
            se_min, se_max = float(self.dataSE.min()), float(self.dataSE.max())
        self.eitMeasurementsSE.axes.set_ylim(se_min, se_max)

        dmin, dmax = float(diff0.min()), float(diff0.max())
        pad = 0.05 * (dmax - dmin + 1e-9)
        self.eitMeasurementsDiff.axes.set_ylim(dmin - pad, dmax + pad)

        # --------- Imagem de reconstrução ---------
        
        # Sempre defina os limites/aspecto antes de desenhar a imagem
        self._lock_axes_extent()

        if method == 'greit':
            self.mySolver.setframes(Vse=self.data[self.frameCounter], method=method)
            self.mySolver.updateImage(self.data[self.frameCounter], method, plot_ref=None)
            img0 = np.real(self.mySolver.image)
            if img0 is None:
                img0 = np.zeros((32, 32))

            # Use o mesmo "extent" do eixo, para GREIT ocupar a mesma área de BP/JAC
            self._plotImage_ref = self.eitImage.axes.imshow(
                img0,
                origin='lower',
                vmin=-0.75, vmax=0.75,
                extent=[self._ax_xlim[0], self._ax_xlim[1], self._ax_ylim[0], self._ax_ylim[1]],
                interpolation='nearest'
            )
            self._attach_or_update_colorbar(self._plotImage_ref)

        elif method in ('jac', 'bp'):
            self.mySolver.setframes(Vse=self.data[self.frameCounter], method=method)
            pts = self.mySolver.mesh_obj.node
            tri = self.mySolver.mesh_obj.element

            if method == 'jac':
                ds_n = sim2pts(pts, tri, np.real(self.mySolver.ds_med_frame))
                self._plotImage_ref = self.eitImage.axes.tripcolor(
                    pts[:, 0], pts[:, 1], tri, ds_n, shading="flat"
                )
            else:  # 'bp'
                c0 = self.mySolver.ds_med_frame
                if c0.ndim > 1:
                    c0 = np.ravel(c0)
                self._plotImage_ref = self.eitImage.axes.tripcolor(
                    pts[:, 0], pts[:, 1], tri, c0
                )

            self._attach_or_update_colorbar(self._plotImage_ref)

        # Títulos iniciais (mantém)
        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")

        self.eitMeasurementsSE.axes.set_title(f"Single-Ended Measurements ({self.frameCounter})")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")

        # Desenha uma vez
        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()
        
    def update_plot(self, data, nframes, method):
        # Dados do frame
        self.dataSE = data[self.frameCounter]
        self.dataDiff = self.mySolver.se_to_diff(self.dataSE)

        # Atualiza curvas SE/Diff
        self._plotSE_ref.set_ydata(self.dataSE)
        self._plotDiff_ref.set_ydata(self.dataDiff)

        
        # Atualiza solução
        self.mySolver.updateImage(self.dataSE, method, self._plotImage_ref)

        # Atualiza imagem conforme método
        if method == 'greit':
            self._plotImage_ref.set_data(np.real(self.mySolver.image))
            # manter limites/extent fixos
            self.eitImage.axes.set_xlim(*self._ax_xlim)
            self.eitImage.axes.set_ylim(*self._ax_ylim)

        elif method == 'jac':
            pts = self.mySolver.mesh_obj.node
            tri = self.mySolver.mesh_obj.element
            ds_n = sim2pts(pts, tri, np.real(self.mySolver.ds_med_frame))
            if hasattr(self._plotImage_ref, 'set_array'):
                self._plotImage_ref.set_array(ds_n)
            else:
                self._plotImage_ref = self.eitImage.axes.tripcolor(
                    pts[:, 0], pts[:, 1], tri, ds_n, shading="flat"
                )
            # manter limites fixos
            self.eitImage.axes.set_xlim(*self._ax_xlim)
            self.eitImage.axes.set_ylim(*self._ax_ylim)

        elif method == 'bp':
            pts = self.mySolver.mesh_obj.node
            tri = self.mySolver.mesh_obj.element
            c = self.mySolver.ds_med_frame
            if c.ndim > 1:
                c = np.ravel(c)
            if hasattr(self._plotImage_ref, 'set_array'):
                self._plotImage_ref.set_array(c)
            else:
                self._plotImage_ref = self.eitImage.axes.tripcolor(
                    pts[:, 0], pts[:, 1], tri, c
                )
            # manter limites fixos
            self.eitImage.axes.set_xlim(*self._ax_xlim)
            self.eitImage.axes.set_ylim(*self._ax_ylim)


        # Atualiza títulos
        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")
        self.eitMeasurementsSE.axes.set_title(f"Single-Ended Measurements ({self.frameCounter})")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")

        # Redesenha
        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()

        # Próximo frame (com wrap)
        self.frameCounter = (self.frameCounter + 1) % nframes

    def on_button_click(window_instance, data, nframes, button='greit'):    
        new_method = button
        print(f"Method set to: {new_method}")
        window_instance.update_solver(new_method)




