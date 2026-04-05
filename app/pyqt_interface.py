import sys
import numpy as np
from PyQt6.QtCore import QSize, QTimer, Qt
from PyQt6.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMessageBox, QGridLayout, QHBoxLayout, QLabel,
    QComboBox, QMainWindow, QPushButton, QTabWidget, QVBoxLayout,
    QWidget, QDoubleSpinBox, QSpinBox, QSlider, QFrame, QScrollArea,
    QSizePolicy, QToolTip
)
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyeit.eit.interp2d import sim2pts
from pyeit_controller import EITsolver
import warnings
import time
import math
import pyeit.mesh.shape as shape


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=2.5, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------

class SectionLabel(QLabel):
    """Small uppercase section header."""
    def __init__(self, text, parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet("""
            QLabel {
                font-size: 10px;
                font-weight: 500;
                color: #888;
                letter-spacing: 1px;
                padding: 10px 14px 4px 14px;
            }
        """)


class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet("color: #e0e0e0; margin: 2px 0;")


class MethodButton(QPushButton):
    """Styled method selection button."""
    ACTIVE_STYLE = """
        QPushButton {
            background: #006633;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 9px 12px;
            font-size: 12px;
            text-align: left;
        }
    """
    INACTIVE_STYLE = """
        QPushButton {
            background: #f5f5f5;
            color: #333;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 9px 12px;
            font-size: 12px;
            text-align: left;
        }
        QPushButton:hover {
            background: #ebebeb;
        }
    """

    def __init__(self, label, tag, parent=None):
        super().__init__(f"{label}   [{tag}]", parent)
        self.setActive(False)

    def setActive(self, active: bool):
        self.setStyleSheet(self.ACTIVE_STYLE if active else self.INACTIVE_STYLE)


class AlgoCard(QWidget):
    """Algorithm description card with chips."""

    ALGO_DATA = {
        'bp': {
            'title': 'Back Projection (BP)',
            'desc': (
                'O BP projeta cada diferença de tensão medida de volta ao domínio, '
                'ao longo das regiões de maior sensibilidade da malha. É o método '
                'mais simples: rápido e intuitivo, mas sem nenhuma regularização — '
                'o que o torna sensível ao ruído presente nos dados.'
            ),
            'chips': [('Sem regularização', 'warn'), ('Sensível a ruído', 'warn'), ('Rápido', 'ok')],
        },
        'greit': {
            'title': 'GREIT',
            'desc': (
                'O GREIT (Generalised Reconstruction algorithm for EIT) calcula uma '
                'matriz de reconstrução linear otimizada durante a configuração. '
                'Produz imagens em grade regular com boa supressão de ruído e '
                'artefatos, sendo amplamente usado em aplicações pulmonares.'
            ),
            'chips': [('Regularização linear', 'ok'), ('Grade uniforme', 'ok'), ('Robusto a ruído', 'ok')],
        },
        'jac': {
            'title': 'Jacobiano (JAC)',
            'desc': (
                'O método Jacobiano (ou Gauss-Newton) resolve iterativamente o '
                'problema inverso usando a matriz de sensibilidade (Jacobiana) '
                'e regularização de Tikhonov. Oferece maior flexibilidade e '
                'precisão, com custo computacional mais elevado.'
            ),
            'chips': [('Regularização Tikhonov', 'ok'), ('Alta precisão', 'ok'), ('Mais lento', 'warn')],
        },
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background: #f9f9f9;
                border: 1px solid #e8e8e8;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_lbl = QLabel()
        self.title_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #006633; border: none; background: transparent;")
        self.title_lbl.setWordWrap(True)

        self.desc_lbl = QLabel()
        self.desc_lbl.setStyleSheet("font-size: 11px; color: #666; line-height: 1.6; border: none; background: transparent;")
        self.desc_lbl.setWordWrap(True)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.desc_lbl)
        
        layout.addStretch(1) #Pushes everything upwards inside "Como Funciona" section

        self.update_method('bp')

    def update_method(self, method: str):
        data = self.ALGO_DATA.get(method, self.ALGO_DATA['bp'])
        self.title_lbl.setText(data['title'])
        self.desc_lbl.setText(data['desc'])


class ParamRow(QWidget):
    """Label + tooltip icon + optional explanation box."""
    def __init__(self, label: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)

        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px; color: #888;")

        tip_btn = QPushButton("?")
        tip_btn.setFixedSize(14, 14)
        tip_btn.setStyleSheet("""
            QPushButton {
                border-radius: 7px;
                border: 1px solid #ccc;
                background: transparent;
                color: #999;
                font-size: 9px;
                padding: 0;
            }
            QPushButton:hover { background: #f0f0f0; }
        """)
        tip_btn.setToolTip(tooltip)
        tip_btn.clicked.connect(lambda: QToolTip.showText(
            tip_btn.mapToGlobal(tip_btn.rect().bottomLeft()), tooltip
        ))

        row.addWidget(lbl)
        row.addWidget(tip_btn)
        row.addStretch()


class StatCard(QWidget):
    def __init__(self, label: str, value: str, parent=None):
        super().__init__(parent)
        # Set background directly via palette — bypasses stylesheet inheritance issues
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('#f5f5f5'))
        self.setPalette(palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet("font-size: 9px; color: #999;")

        self._val = QLabel(value)
        self._val.setStyleSheet("font-size: 16px; font-weight: 500; color: #222;")

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)

    def setValue(self, v: str):
        self._val.setText(v)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, data, nframes, method='greit'):
        self.data = data
        self.nframes = nframes
        self.method = method
        self._colorbar_ref = None
        self._greit_extent = None

        super().__init__()

        self.setWindowTitle("EITduino")
        self.setMinimumSize(QSize(1100, 680))

        # ------------------------------------------------------------------
        # Root layout: three columns
        # ------------------------------------------------------------------
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # ---- LEFT PANEL --------------------------------------------------
        left = QWidget()
        left.setFixedWidth(280)
        left.setStyleSheet("background: white; border-right: 1px solid #e8e8e8;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Header / branding
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background: #006633;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 0, 14, 0)

        logo_lbl = QLabel()
        pix = QPixmap('images/logo_UFABC.png')
        if not pix.isNull():
            logo_lbl.setPixmap(pix.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

        title_lbl = QLabel("EITduino")
        title_lbl.setStyleSheet("color: white; font-size: 18px; font-weight: 500;")

        header_layout.addWidget(logo_lbl)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        left_layout.addWidget(header)

        # Method selection
        left_layout.addWidget(SectionLabel("Método de Reconstrução"))

        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(12, 0, 12, 0)
        btn_layout.setSpacing(5)

        self.btn_bp    = MethodButton("Back Projection", "BP")
        self.btn_greit = MethodButton("GREIT",           "GR")
        self.btn_jac   = MethodButton("Jacobiano",       "JAC")

        self.btn_bp.clicked.connect(lambda: self.update_solver('bp'))
        self.btn_greit.clicked.connect(lambda: self.update_solver('greit'))
        self.btn_jac.clicked.connect(lambda: self.update_solver('jac'))

        btn_layout.addWidget(self.btn_bp)
        btn_layout.addWidget(self.btn_greit)
        btn_layout.addWidget(self.btn_jac)
        left_layout.addWidget(btn_container)
        left_layout.addWidget(Divider())

        # Algorithm description card
        left_layout.addWidget(SectionLabel("Como funciona"))
        self.algo_card = AlgoCard()
        card_wrapper = QWidget()
        cw_layout = QVBoxLayout(card_wrapper)
        cw_layout.setContentsMargins(12, 0, 12, 8)
        cw_layout.addWidget(self.algo_card)
        card_wrapper.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.algo_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        left_layout.addWidget(card_wrapper, stretch=1)
        left_layout.addWidget(Divider())

        left_layout.addWidget(Divider())

        # ---- CENTER PANEL ------------------------------------------------
        center = QWidget()
        center.setStyleSheet("background: white;")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.eitImage = MplCanvas(self)
        self._colorbar_ref = None
        self._cbar_ax = None
        self._ax_xlim = None
        self._ax_ylim = None
        self._raster_cache = None
        self._grid_extent = None
        self._nx = self._ny = 128

        try:
            self.eitImage.fig.set_constrained_layout(False)
        except Exception:
            pass

        center_layout.addWidget(self.eitImage, stretch=1)

        # Measurement strip
        meas_strip = QWidget()
        meas_strip.setFixedHeight(100)
        meas_strip.setStyleSheet("border-top: 1px solid #e8e8e8;")
        meas_layout = QHBoxLayout(meas_strip)
        meas_layout.setContentsMargins(0, 0, 0, 0)
        meas_layout.setSpacing(0)

        se_widget = QWidget()
        se_widget.setStyleSheet("border-right: 1px solid #e8e8e8;")
        se_layout = QVBoxLayout(se_widget)
        se_layout.setContentsMargins(10, 6, 10, 6)
        self._se_title = QLabel("Medições Single-Ended")
        self._se_title.setStyleSheet("font-size: 10px; color: #999;")
        self.eitMeasurementsSE = MplCanvas(self, width=6, height=1.2)
        self.eitMeasurementsSE.fig.patch.set_alpha(0)
        se_layout.addWidget(self._se_title)
        se_layout.addWidget(self.eitMeasurementsSE)

        diff_widget = QWidget()
        diff_layout = QVBoxLayout(diff_widget)
        diff_layout.setContentsMargins(10, 6, 10, 6)
        self._diff_title = QLabel("Medições Diferenciais")
        self._diff_title.setStyleSheet("font-size: 10px; color: #999;")
        self.eitMeasurementsDiff = MplCanvas(self, width=6, height=1.2)
        self.eitMeasurementsDiff.fig.patch.set_alpha(0)
        diff_layout.addWidget(self._diff_title)
        diff_layout.addWidget(self.eitMeasurementsDiff)

        meas_layout.addWidget(se_widget)
        meas_layout.addWidget(diff_widget)
        center_layout.addWidget(meas_strip)

        # ---- RIGHT PANEL -------------------------------------------------
        right_scroll = QScrollArea()
        right_scroll.setFixedWidth(272)
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; border-left: 1px solid #e8e8e8; background: white; }")

        right = QWidget()
        right.setStyleSheet("background: white;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 12)
        right_layout.setSpacing(0)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_scroll.setWidget(right)

        # -- Colormap
        right_layout.addWidget(SectionLabel("Controles"))

        def _ctrl_block(label, tooltip, widget):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(12, 0, 12, 8)
            l.setSpacing(4)
            l.addWidget(ParamRow(label, tooltip))
            l.addWidget(widget)
            return w

        self.cmbCmap = QComboBox()
        self.cmbCmap.addItems(["RdBu_r", "viridis", "plasma", "inferno", "CET-D1"])
        self.cmbCmap.setStyleSheet("font-size: 11px; background: white; color: #333;")
        right_layout.addWidget(_ctrl_block("Colormap",
            "Paleta de cores usada para representar variações de condutividade na imagem.",
            self.cmbCmap))

        # -- Levels
        levels_widget = QWidget()
        lvl_layout = QHBoxLayout(levels_widget)
        lvl_layout.setContentsMargins(0, 0, 0, 0)
        lvl_layout.setSpacing(6)
        self.spnVmin = QDoubleSpinBox()
        self.spnVmin.setDecimals(3); self.spnVmin.setRange(-1e6, 1e6); self.spnVmin.setValue(-1.0)
        self.spnVmin.setStyleSheet("font-size: 11px; background: white; color: #333;")
        self.spnVmax = QDoubleSpinBox()
        self.spnVmax.setDecimals(3); self.spnVmax.setRange(-1e6, 1e6); self.spnVmax.setValue(1.0)
        self.spnVmax.setStyleSheet("font-size: 11px; background: white; color: #333;")
        lvl_layout.addWidget(self.spnVmin)
        lvl_layout.addWidget(self.spnVmax)
        right_layout.addWidget(_ctrl_block("Níveis (min / max)",
            "Define os limites do mapa de cores. Valores fora do intervalo ficam saturados na cor extrema.",
            levels_widget))

        # -- BP smoothing slider
        alpha_widget = QWidget()
        alpha_layout = QHBoxLayout(alpha_widget)
        alpha_layout.setContentsMargins(0, 0, 0, 0)
        alpha_layout.setSpacing(8)
        self.sldAlpha = QSlider(Qt.Orientation.Horizontal)
        self.sldAlpha.setRange(0, 95); self.sldAlpha.setSingleStep(5); self.sldAlpha.setValue(40)
        self.lblAlpha = QLabel("α = 0.40")
        self.lblAlpha.setStyleSheet("font-size: 10px; color: #888; min-width: 54px;")
        self.sldAlpha.valueChanged.connect(self._on_alpha_changed)
        alpha_layout.addWidget(self.sldAlpha)
        alpha_layout.addWidget(self.lblAlpha)
        right_layout.addWidget(_ctrl_block("Suavização BP (α)",
            "Mistura temporal entre frames consecutivos. α = 0: sem suavização. α = 0.9: suavização intensa, mas com atraso visual.",
            alpha_widget))

        # -- Grid resolution slider
        res_widget = QWidget()
        res_layout = QHBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(8)
        self.sldRes = QSlider(Qt.Orientation.Horizontal)
        self.sldRes.setRange(32, 256); self.sldRes.setSingleStep(16); self.sldRes.setValue(128)
        self.lblRes = QLabel("128 px")
        self.lblRes.setStyleSheet("font-size: 10px; color: #888; min-width: 40px;")
        
        res_layout.addWidget(self.sldRes)
        res_layout.addWidget(self.lblRes)
        right_layout.addWidget(_ctrl_block("Resolução da grade",
            "Número de pixels por eixo na imagem interpolada. Valores maiores produzem imagens mais suaves, mas reduzem o FPS.",
            res_widget))
        
        # -- Frame selector
        frame_widget = QWidget()
        frame_layout = QVBoxLayout(frame_widget)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(4)

        # Play/pause button + frame label on same row
        playrow = QHBoxLayout()
        self.btnPlayPause = QPushButton("⏸ Pausar")
        self.btnPlayPause.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                border-radius: 6px;
                border: 1px solid #ccc;
                background: #f5f5f5;
                font-size: 11px;
                color: #555;
            }
            QPushButton:hover { background: #ebebeb; }
        """)
        self.btnPlayPause.clicked.connect(self._toggle_play_pause)
        self._is_playing = True

        self.lblFramePos = QLabel("0 / 0")
        self.lblFramePos.setStyleSheet("font-size: 10px; color: #888;")

        playrow.addWidget(self.btnPlayPause)
        playrow.addStretch()
        playrow.addWidget(self.lblFramePos)

        self.sldFrame = QSlider(Qt.Orientation.Horizontal)
        self.sldFrame.setRange(0, self.nframes - 1)
        self.sldFrame.setValue(0)
        self.sldFrame.valueChanged.connect(self._on_frame_scrub)

        frame_layout.addLayout(playrow)
        frame_layout.addWidget(self.sldFrame)

        right_layout.addWidget(_ctrl_block(
            "Navegação de frames",
            "Pause a animação e arraste para inspecionar frames específicos. Útil para comparar a reconstrução em momentos distintos.",
            frame_widget))
        
        # -- Animation speed
        speed_widget = QWidget()
        speed_layout = QHBoxLayout(speed_widget)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(8)
        self.sldSpeed = QSlider(Qt.Orientation.Horizontal)
        self.sldSpeed.setRange(1, 20)    # FPS: 1 to 20
        self.sldSpeed.setSingleStep(1)
        self.sldSpeed.setValue(10)       # default 10 FPS
        self.lblSpeed = QLabel("10 FPS")
        self.lblSpeed.setStyleSheet("font-size: 10px; color: #888; min-width: 44px;")
        self.sldSpeed.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.sldSpeed)
        speed_layout.addWidget(self.lblSpeed)
        right_layout.addWidget(_ctrl_block("Velocidade (frames/s)",
            "Controla a velocidade de reprodução dos dados gravados.",
            speed_widget))

        # Parameters section
        right_layout.addWidget(SectionLabel("Parâmetros da malha"))

        params_container = QWidget()
        params_layout = QVBoxLayout(params_container)
        params_layout.setContentsMargins(12, 0, 12, 8)
        params_layout.setSpacing(8)

        params_layout.addWidget(ParamRow("Tipo de regularização (JAC)",
            "Método de regularização usado pelo Jacobiano. Kotre é o padrão. Só se aplica ao JAC."))
        self.cmbRegType = QComboBox()
        self.cmbRegType.addItems(["kotre", "lm"])
        self.cmbRegType.setStyleSheet("font-size: 11px; background: white; color: #333;")
        self.cmbRegType.currentTextChanged.connect(self._on_reg_type_changed)
        params_layout.addWidget(self.cmbRegType)

        # Shape selector
        params_layout.addWidget(ParamRow("Formato da malha",
            "Define a geometria do domínio. Círculo é o padrão para a maioria dos experimentos. Elipse pode modelar geometrias não simétricas."))
        self.cmbShape = QComboBox()
        self.cmbShape.addItems(["Círculo", "Elipse"])
        self.cmbShape.setStyleSheet("font-size: 11px; background: white; color: #333;")
        params_layout.addWidget(self.cmbShape)

        # h0
        params_layout.addWidget(ParamRow("Densidade da malha (h0)",
            "Tamanho dos triângulos. Menor = mais triângulos, mais preciso, mais lento. Seguro: 0.04 a 0.20."))
        h0_row = QHBoxLayout()
        self.spnH0 = QDoubleSpinBox()
        self.spnH0.setDecimals(3); self.spnH0.setRange(0.04, 0.20)
        self.spnH0.setSingleStep(0.01); self.spnH0.setValue(0.1)
        self.spnH0.setStyleSheet("font-size: 11px; background: white; color: #333;")
        self.lblTriEstimate = QLabel()
        self.lblTriEstimate.setStyleSheet("font-size: 10px; color: #888;")
        self.spnH0.valueChanged.connect(self._update_tri_estimate)
        h0_row.addWidget(self.spnH0)
        h0_row.addWidget(self.lblTriEstimate)
        params_layout.addLayout(h0_row)

        # Lamb
        params_layout.addWidget(ParamRow("Regularização (λ)",
            "Força da regularização para JAC e GREIT. Maior = mais suave, menos sensível. Não se aplica ao BP."))
        self.spnLamb = QDoubleSpinBox()
        self.spnLamb.setDecimals(4); self.spnLamb.setRange(0.0001, 1.0)
        self.spnLamb.setSingleStep(0.001); self.spnLamb.setValue(0.01)
        self.spnLamb.setStyleSheet("font-size: 11px; background: white; color: #333;")
        self.spnLamb.valueChanged.connect(self._on_lamb_changed)
        params_layout.addWidget(self.spnLamb)

        right_layout.addWidget(params_container)
        right_layout.addWidget(Divider())

        # -- Action buttons
        def _action_btn(text, callback, green=False):
            btn = QPushButton(text)
            style = """
                QPushButton {{
                    margin: 0 12px;
                    padding: 7px;
                    border-radius: 6px;
                    border: 1px solid {border};
                    background: {bg};
                    color: {fg};
                    font-size: 11px;
                }}
                QPushButton:hover {{ background: {hover}; }}
            """
            if green:
                btn.setStyleSheet(style.format(border="#006633", bg="transparent", fg="#006633", hover="#e8f5ee"))
            else:
                btn.setStyleSheet(style.format(border="#ccc", bg="#f5f5f5", fg="#555", hover="#ebebeb"))
            btn.clicked.connect(callback)
            return btn

        btnApply   = _action_btn("Aplicar colormap / níveis", self._apply_levels_and_cmap, green=True)
        btnRebuild = _action_btn("Reconstruir malha",         self._rebuild_mesh_from_controls)
        btnSwitch  = _action_btn("Alternar para PyQtGraph",   self.switch_to_pyqtgraph)

        for btn in [btnApply, btnRebuild, btnSwitch]:
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 3, 0, 3)
            wl.addWidget(btn)
            right_layout.addWidget(wrapper)

        right_layout.addWidget(Divider())

        # -- Assemble root
        root_layout.addWidget(left)
        root_layout.addWidget(center, stretch=1)
        root_layout.addWidget(right_scroll)

        # ------------------------------------------------------------------
        # Solver init
        # ------------------------------------------------------------------
        self.mySolver = EITsolver(method=method, h0=0.1, fd = shape.circle)
        self._plotImage_ref = None
        self._plotSE_ref    = None
        self._plotDiff_ref  = None
        self.frameCounter   = 0
        self._last_t        = None
        self._fps_alpha     = 0.9
        self._fps_est       = 0.0
        self._comparison_win = None

        self.mySolver.setVref(self.data[0])
        self._update_method_ui(method)

        self.init_plots(data=data, method=method)
        self.update_plot(data, nframes, method=method)

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    # ------------------------------------------------------------------
    # Raster cache
    # ------------------------------------------------------------------

    def _build_raster_cache(self):
        pts = self.mySolver.mesh_obj.node
        tri = self.mySolver.mesh_obj.element
        x_min, x_max = float(pts[:,0].min()), float(pts[:,0].max())
        y_min, y_max = float(pts[:,1].min()), float(pts[:,1].max())
        self._grid_extent = [x_min, x_max, y_min, y_max]
        xs = np.linspace(x_min, x_max, self._nx)
        ys = np.linspace(y_min, y_max, self._ny)
        gx, gy = np.meshgrid(xs, ys)
        triang = mtri.Triangulation(pts[:,0], pts[:,1], tri)
        X, Y = gx.ravel(), gy.ravel()
        finder = triang.get_trifinder()
        tri_idx = finder(X, Y)
        n = tri_idx.size
        v0=np.full(n,-1,dtype=int); v1=v0.copy(); v2=v0.copy()
        w0=np.zeros(n); w1=w0.copy(); w2=w0.copy()
        valid = tri_idx >= 0
        if np.any(valid):
            idxs = tri_idx[valid]
            for t in np.unique(idxs):
                pix = np.nonzero(valid)[0][idxs==t]
                T=tri[t]; A,B,C=pts[T[0]],pts[T[1]],pts[T[2]]
                Px,Py=X[pix],Y[pix]
                den=(B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1])
                if den==0.: continue
                l0=((B[1]-C[1])*(Px-C[0])+(C[0]-B[0])*(Py-C[1]))/den
                l1=((C[1]-A[1])*(Px-C[0])+(A[0]-C[0])*(Py-C[1]))/den
                l2=1.-l0-l1
                v0[pix]=T[0];v1[pix]=T[1];v2[pix]=T[2]
                w0[pix]=l0; w1[pix]=l1; w2[pix]=l2
        self._raster_cache={"v0":v0,"v1":v1,"v2":v2,
                             "w0":w0,"w1":w1,"w2":w2,"mask":valid}

    def _rasterize(self, nodal):
        c=self._raster_cache; valid=c["mask"]
        flat=np.full(c["v0"].shape,np.nan)
        idx=np.nonzero(valid)[0]
        flat[idx]=(c["w0"][idx]*nodal[c["v0"][idx]]+
                   c["w1"][idx]*nodal[c["v1"][idx]]+
                   c["w2"][idx]*nodal[c["v2"][idx]])
        return np.ma.masked_invalid(flat.reshape(self._ny,self._nx))

    def _mesh_to_nodal(self, vals):
        pts=self.mySolver.mesh_obj.node; tri=self.mySolver.mesh_obj.element
        v=np.real(vals)
        if v.ndim>1: v=np.ravel(v)
        if v.shape[0]==pts.shape[0]: return v
        elif v.shape[0]==tri.shape[0]: return sim2pts(pts,tri,v)
        return np.zeros(pts.shape[0])

    def _rasterize_method(self, method):
        nodal = self._mesh_to_nodal(
            self.mySolver.image if method=='bp' else self.mySolver.ds_med_frame
        )
        return self._rasterize(nodal)
    
    def _on_lamb_changed(self, value: float):
        """Updates regularization and reruns setup — no mesh rebuild needed."""
        self.mySolver.hp['lamb'] = value
        if self.method in ('greit', 'jac'):
            self.mySolver.setup()

    def _toggle_play_pause(self):
        """Toggles animation on/off."""
        if self._is_playing:
            self.timer.stop()
            self._is_playing = False
            self.btnPlayPause.setText("▶ Retomar")
        else:
            self.timer.start()
            self._is_playing = True
            self.btnPlayPause.setText("⏸ Pausar")

    def _on_frame_scrub(self, value: int):
        """When slider is moved manually, jump to that frame."""
        # Only act if the user is scrubbing (not just the slider following playback)
        if not self._is_playing:
            self.frameCounter = value
            self.update_plot(self.data, self.nframes, method=self.method)

    # ------------------------------------------------------------------
    # Electrode overlay
    # ------------------------------------------------------------------

    def _draw_electrode_overlay(self):
        """Draw numbered electrode markers on the image axes."""
        ax = self.eitImage.axes
        mesh = self.mySolver.mesh_obj
        pts  = mesh.node

        # el_pos contains the node indices of the electrodes
        el_pos = getattr(mesh, 'el_pos', None)
        if el_pos is None:
            return

        for i, node_idx in enumerate(el_pos):
            x, y = pts[node_idx, 0], pts[node_idx, 1]
            ax.plot(x, y, 'o',
                    markersize=9,
                    color='#006633',
                    markeredgecolor='white',
                    markeredgewidth=1.2,
                    zorder=5)
            ax.text(x, y, str(i + 1),
                    ha='center', va='center',
                    fontsize=6, color='white',
                    fontweight='bold', zorder=6)

    # ------------------------------------------------------------------
    # Axes helpers
    # ------------------------------------------------------------------

    def _lock_axes_extent(self):
        pts=self.mySolver.mesh_obj.node
        x_min,x_max=float(pts[:,0].min()),float(pts[:,0].max())
        y_min,y_max=float(pts[:,1].min()),float(pts[:,1].max())
        span=max(x_max-x_min,y_max-y_min)
        margin=0.05*span
        self._ax_xlim=(x_min-margin,x_max+margin)
        self._ax_ylim=(y_min-margin,y_max+margin)
        ax=self.eitImage.axes
        ax.set_xlim(*self._ax_xlim)
        ax.set_ylim(*self._ax_ylim)
        ax.set_aspect('equal',adjustable='box')
        ax.axis('off')

    def _attach_or_update_colorbar(self, mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if self._colorbar_ref is None or self._cbar_ax is None:
            divider=make_axes_locatable(self.eitImage.axes)
            self._cbar_ax=divider.append_axes("right",size="4%",pad=0.05)
            self._colorbar_ref=self.eitImage.fig.colorbar(mappable,cax=self._cbar_ax)
            self._cbar_ax.set_ylabel("Δσ (norm.)", fontsize=7, labelpad=3)
            self._cbar_ax.tick_params(labelsize=7)
        else:
            self._colorbar_ref.update_normal(mappable)

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _update_method_ui(self, method: str):
        self.btn_bp.setActive(method == 'bp')
        self.btn_greit.setActive(method == 'greit')
        self.btn_jac.setActive(method == 'jac')
        self.algo_card.update_method(method)

        self.spnLamb.setEnabled(method in ('greit', 'jac'))
        self.spnLamb.setStyleSheet(
            "font-size: 11px; background: white; color: #333;"
            if method in ('greit', 'jac') else
            "font-size: 11px; background: #f0f0f0; color: #aaa;"
        )
        # Reg type only applies to JAC
        self.cmbRegType.setEnabled(method == 'jac')
        self.cmbRegType.setStyleSheet(
            "font-size: 11px; background: white; color: #333;"
            if method == 'jac' else
            "font-size: 11px; background: #f0f0f0; color: #aaa;"
    )

    def _on_reg_type_changed(self, value: str):
        self.mySolver.hp['jac_method'] = value
        if self.method == 'jac':
            self.mySolver.setup()


    def _update_tri_estimate(self):
        """Shows actual count if h0 matches current mesh, otherwise prompts rebuild."""
        h0 = self.spnH0.value()
        
        # Guard: solver may not exist yet during UI construction
        if not hasattr(self, 'mySolver'):
            self.lblTriEstimate.setText("→ reconstruir para atualizar")
            return
        
        current_h0 = self.mySolver.h0
        if abs(h0 - current_h0) < 1e-6:
            actual = self.mySolver.mesh_obj.element.shape[0]
            self.lblTriEstimate.setText(f"= {actual} triângulos (atual)")
        else:
            self.lblTriEstimate.setText("→ reconstruir para atualizar")

    def _on_alpha_changed(self, value):
        alpha = value / 100.0
        self.lblAlpha.setText(f"α = {alpha:.2f}")
        self.mySolver.bp_temporal_alpha = alpha

    # ------------------------------------------------------------------
    # Solver switching
    # ------------------------------------------------------------------

    def update_solver(self, new_method: str):
        if new_method == self.method:
            return
        try: self.timer.stop()
        except Exception: pass

        self.method = new_method
        self._update_method_ui(new_method)

        self.eitImage.axes.clear()
        self.eitMeasurementsSE.axes.clear()
        self.eitMeasurementsDiff.axes.clear()

        if self._colorbar_ref is not None:
            try: self._colorbar_ref.remove()
            except Exception: pass
            self._colorbar_ref = None
            self._cbar_ax = None

        fd_map = {"Círculo": shape.circle, "Elipse": shape.ellipse}
        selected_fd = fd_map.get(self.cmbShape.currentText(), shape.circle)
        self.mySolver.recreate_mesh(method=self.method, h0=self.spnH0.value(), fd=selected_fd)
        self.mySolver.setVref(self.data[0])

        self._plotImage_ref = None
        self._plotSE_ref    = None
        self._plotDiff_ref  = None
        self.frameCounter   = 0

        self._nx = self._ny = self.sldRes.value()
        self._raster_cache = None
        self._build_raster_cache()

        self.init_plots(data=self.data, method=self.method)
        self.update_plot(self.data, self.nframes, method=self.method)
        self.timer.start()

    # ------------------------------------------------------------------
    # Plot init
    # ------------------------------------------------------------------

    def init_plots(self, data, method='greit'):
        self.frameCounter = 0
        self.dataSE = data[0]

        self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE, lw=1, color='#1f77b4')[0]
        diff0 = self.mySolver.se_to_diff(self.dataSE)
        self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(diff0, lw=1, color='#ff7f0e')[0]

        for canvas, ax in [(self.eitMeasurementsSE, self.eitMeasurementsSE.axes),
                           (self.eitMeasurementsDiff, self.eitMeasurementsDiff.axes)]:
            ax.set_facecolor('none')
            ax.tick_params(labelsize=7)
            canvas.fig.patch.set_alpha(0)

        try:
            se_min,se_max=float(self.data.min()),float(self.data.max())
        except Exception:
            se_min,se_max=float(self.dataSE.min()),float(self.dataSE.max())
        self.eitMeasurementsSE.axes.set_ylim(se_min,se_max)

        dmin,dmax=float(diff0.min()),float(diff0.max())
        pad=0.05*(dmax-dmin+1e-9)
        self.eitMeasurementsDiff.axes.set_ylim(dmin-pad,dmax+pad)

        if self._raster_cache is None:
            self._nx=self._ny=128
            self._build_raster_cache()

        self._lock_axes_extent()
        self.mySolver.updateImage(data[self.frameCounter], method, plot_ref=None)

        vmin,vmax=float(self.spnVmin.value()),float(self.spnVmax.value())
        cmap=self.cmbCmap.currentText()

        if method == 'greit':
            self.mySolver.updateImage(data[self.frameCounter], method, plot_ref=None)
            img0 = np.real(self.mySolver.image)

            # Get actual GREIT grid extent from mask_value
            x_greit, y_greit, _ = self.mySolver.eit.mask_value(
                self.mySolver.ds_med_frame, mask_value=np.nan
            )
            self._greit_extent = [
                float(x_greit.min()), float(x_greit.max()),
                float(y_greit.min()), float(y_greit.max())
            ]

            _cmap = plt.get_cmap(cmap).copy()
            _cmap.set_bad(alpha=0)  # NaN pixels transparent

            self._plotImage_ref = self.eitImage.axes.imshow(
                img0, origin='lower', vmin=vmin, vmax=vmax, cmap=_cmap,
                extent=self._greit_extent,
                interpolation='bilinear'
            )

        elif method in ('bp','jac'):
            img0=self._rasterize_method(method)
            self._plotImage_ref=self.eitImage.axes.imshow(
                img0,origin='lower',vmin=vmin,vmax=vmax,cmap=cmap,
                extent=[self._grid_extent[0],self._grid_extent[1],
                        self._grid_extent[2],self._grid_extent[3]],
                interpolation='bilinear')

        self._attach_or_update_colorbar(self._plotImage_ref)
        self._draw_electrode_overlay()

        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def _on_timer(self):
        self.update_plot(self.data, self.nframes, method=self.method)

    def update_plot(self, data, nframes, method):
        if not self.isVisible():
            return

        self.dataSE   = data[self.frameCounter]
        self.dataDiff = self.mySolver.se_to_diff(self.dataSE)

        self._plotSE_ref.set_ydata(self.dataSE)
        self._plotDiff_ref.set_ydata(self.dataDiff)

        self.mySolver.updateImage(self.dataSE, method, plot_ref=None)

        if method=='greit':
            self._plotImage_ref.set_data(np.real(self.mySolver.image))
        elif method in ('bp','jac'):
            img=self._rasterize_method(method)
            self._plotImage_ref.set_data(img)
            self._plotImage_ref.set_clim(-1.0,1.0)

        self.eitImage.axes.set_xlim(*self._ax_xlim)
        self.eitImage.axes.set_ylim(*self._ax_ylim)

        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()

        # Update frame slider and label to follow playback
        self.sldFrame.blockSignals(True)   # prevent _on_frame_scrub from firing
        self.sldFrame.setValue(self.frameCounter)
        self.sldFrame.blockSignals(False)
        self.lblFramePos.setText(f"{self.frameCounter} / {self.nframes - 1}")

        self.frameCounter = (self.frameCounter + 1) % nframes

        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps_est = self._fps_alpha*self._fps_est + (1-self._fps_alpha)*inst
                try:
                    self.setWindowTitle(
                        f"EITduino  —  {self.method.upper()}  |  FPS: {self._fps_est:.1f}  |  Frame {self.frameCounter}"
                    )
                except Exception:
                    pass
        self._last_t = t
    
    def _on_speed_changed(self, fps: int):
        self.lblSpeed.setText(f"{fps} FPS")
        self.timer.setInterval(1000 // fps)
    
    def _on_res_changed(self, value: int):
        self.lblRes.setText(f"{value} px")
        if self.method == 'greit':
            # For GREIT, resolution controls the reconstruction grid
            self.mySolver.hp['greit_n'] = value // 4  # map px to grid size
            self.mySolver.setup()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _apply_levels_and_cmap(self):
        cmap=self.cmbCmap.currentText()
        vmin=float(self.spnVmin.value())
        vmax=float(self.spnVmax.value())
        try:
            self._plotImage_ref.set_cmap(cmap)
            self._plotImage_ref.set_clim(vmin,vmax)
        except Exception:
            pass
        self.eitImage.draw()

    def _rebuild_mesh_from_controls(self):
        res = self.sldRes.value()
        h0 = self.spnH0.value()   # ← use user-specified h0

        # Still try a small fallback in case the chosen h0 causes instability
        h0_trials = [h0, h0 * 0.9, h0 * 0.8]
        success = False
        last_err = None

        fd_map = {"Círculo": shape.circle, "Elipse": shape.ellipse}
        selected_fd = fd_map.get(self.cmbShape.currentText(), shape.circle)

        for h0_try in h0_trials:
            try:
                self.mySolver.recreate_mesh(
                    n_el=self.mySolver.n_el, fd=selected_fd,
                    method=self.method, h0=h0_try)
                
                self._on_res_changed(res)
                self.mySolver.setVref(self.data[0])
                
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    self.mySolver.setframes(self.data[0], self.method)
                success = True
                break
            except Exception as e:
                last_err = e
                continue

        if success:
            self._nx = self._ny = res
            self._raster_cache = None
            self._build_raster_cache()
            self.eitImage.axes.clear()
            self.eitMeasurementsSE.axes.clear()
            self.eitMeasurementsDiff.axes.clear()
            if self._colorbar_ref is not None:
                try: self._colorbar_ref.remove()
                except Exception: pass
                self._colorbar_ref = None
                self._cbar_ax = None
            self._plotImage_ref = None
            self._plotSE_ref    = None
            self._plotDiff_ref  = None
            self.frameCounter   = 0
            self._update_tri_estimate()
            self.init_plots(data=self.data, method=self.method)
            self.update_plot(self.data, self.nframes, method=self.method)
        else:
            QMessageBox.warning(self, "Reconstruir malha",
                                f"Falha ao reconstruir com h0={h0:.3f}.\n"
                                f"Tente um valor maior.\nDetalhe: {last_err}")

    # ------------------------------------------------------------------
    # Comparison window (stub for Phase 2)
    # ------------------------------------------------------------------

    def _open_comparison(self):
        QMessageBox.information(self, "Comparação",
            "A janela de comparação de métodos será implementada na Fase 2.")

    # ------------------------------------------------------------------
    # Switch interface
    # ------------------------------------------------------------------

    def switch_to_pyqtgraph(self):
        try: self._dispose_matplotlib()
        except Exception: pass
        def _launch():
            from pyqtgraph_interface import MainWindowPG
            app=QApplication.instance()
            new_win=MainWindowPG(self.data,self.nframes,method=self.method)
            app.setProperty('active_window',new_win)
            new_win.show()
        QTimer.singleShot(0,_launch)
        self.close()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _dispose_matplotlib(self):
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._on_timer)
        except Exception: pass
        if self._colorbar_ref is not None:
            try: self._colorbar_ref.remove()
            except Exception: pass
            self._colorbar_ref=None; self._cbar_ax=None
        for canvas in [self.eitImage,self.eitMeasurementsSE,self.eitMeasurementsDiff]:
            try: canvas.axes.cla(); canvas.draw()
            except Exception: pass
            try: canvas.setParent(None); canvas.deleteLater()
            except Exception: pass
        try:
            plt.close(self.eitImage.fig)
            plt.close(self.eitMeasurementsSE.fig)
            plt.close(self.eitMeasurementsDiff.fig)
        except Exception: pass

    def closeEvent(self, event):
        try: self._dispose_matplotlib()
        except Exception: pass
        super().closeEvent(event)