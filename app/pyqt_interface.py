import sys
import numpy as np
from PyQt6.QtCore import QSize, QTimer
from PyQt6.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMessageBox, QGridLayout, QHBoxLayout, QLabel,
    QComboBox, QMainWindow, QPushButton, QTabWidget, QVBoxLayout,
    QWidget, QDoubleSpinBox, QSpinBox
)
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from pyeit.eit.interp2d import sim2pts
from pyeit_controller import EITsolver
import warnings
import time


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

    def __init__(self, data, nframes, method='greit'):
        self.data = data
        self.nframes = nframes
        self.method = method
        self._colorbar_ref = None

        super(MainWindow, self).__init__()

        self.setWindowTitle("EITduino")
        self.setMinimumSize(QSize(800, 600))
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('white'))
        self.setPalette(palette)

        # Header
        layout_header = QGridLayout()
        layout_header.setColumnStretch(0, 1)
        layout_header.setColumnStretch(1, 10)

        logoUFABC = QLabel("")
        pix = QPixmap('images/logo_UFABC.png')
        if not pix.isNull():
            logoUFABC.setPixmap(pix.scaled(75, 75))
        else:
            print("Warning: logo_UFABC.png not found or invalid.")

        headerText = QLabel("EITduino")
        headerText.setStyleSheet("QLabel { color : #006633; font size : 40; }")
        fHeader = QFont("Humanst777", 50, weight=625)
        headerText.setFont(fHeader)

        layout_header.addWidget(logoUFABC, 0, 0)
        layout_header.addWidget(headerText, 0, 1)

        # GUI layout
        self.layout_gui = QGridLayout()
        self.layout_gui.setColumnStretch(0, 1)
        self.layout_gui.setColumnMinimumWidth(0, 400)
        self.layout_gui.setColumnStretch(1, 10)

        self.eitMeasurementsSE = MplCanvas(self, width=10, height=2)
        self.eitMeasurementsDiff = MplCanvas(self, width=10, height=2)

        layout_measurements = QVBoxLayout()
        layout_measurements.addWidget(self.eitMeasurementsSE)
        layout_measurements.addWidget(self.eitMeasurementsDiff)
        measurements_widget = QWidget()
        measurements_widget.setLayout(layout_measurements)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { background: #222529; padding: 8px; }
            QTabBar::tab:selected { background: #006633; color: white; font-weight: bold; }
        """)

        # Solver Config tab
        tabConfig = QWidget()
        tabConfig_layout = QVBoxLayout()
        tabConfig_label = Color('lightgray')
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

        self.tabs.addTab(tabConfig, 'Solver Config')
        self.tabs.addTab(measurements_widget, 'Measurements')

        self.eitImage = MplCanvas(self)
        self._colorbar_ref = None
        self._cbar_ax = None
        self._ax_xlim = None
        self._ax_ylim = None

        # Grid raster cache — same strategy as PyQtGraph interface
        self._raster_cache = None
        self._grid_extent = None
        self._nx = self._ny = 128

        try:
            self.eitImage.fig.set_constrained_layout(False)
        except Exception:
            pass

        self.layout_gui.addWidget(self.tabs)
        self.layout_gui.addWidget(self.eitImage, 0, 1)

        layout_main = QGridLayout()
        layout_main.setRowStretch(0, 1)
        layout_main.setRowStretch(1, 10)
        layout_main.addLayout(layout_header, 0, 0)
        layout_main.addLayout(self.layout_gui, 1, 0)

        main_widget = QWidget()
        main_widget.setLayout(layout_main)
        self.setCentralWidget(main_widget)

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

        # Solver — setVref BEFORE init_plots
        self.mySolver = EITsolver(method=method, h0=0.1)
        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0

        self.mySolver.setVref(self.data[0])

        self.init_plots(data=data, method=method)
        self.update_plot(data, nframes, method=method)

        # Controls tab
        tabControls = QWidget()
        vbox_ctrl = QVBoxLayout(tabControls)
        vbox_ctrl.setSpacing(8)

        vbox_ctrl.addWidget(QLabel("Colormap"))
        self.cmbCmap = QComboBox()
        self.cmbCmap.addItems(["RdBu_r", "viridis", "plasma", "inferno", "CET-D1"])
        vbox_ctrl.addWidget(self.cmbCmap)

        vbox_ctrl.addWidget(QLabel("Levels (min / max)"))
        row_levels = QHBoxLayout()
        self.spnVmin = QDoubleSpinBox()
        self.spnVmin.setDecimals(3); self.spnVmin.setRange(-1e6, 1e6); self.spnVmin.setValue(-1.0)
        self.spnVmax = QDoubleSpinBox()
        self.spnVmax.setDecimals(3); self.spnVmax.setRange(-1e6, 1e6); self.spnVmax.setValue(1.0)
        row_levels.addWidget(self.spnVmin); row_levels.addWidget(self.spnVmax)
        vbox_ctrl.addLayout(row_levels)

        vbox_ctrl.addWidget(QLabel("Grid resolution (nx = ny)"))
        self.spnRes = QSpinBox()
        self.spnRes.setRange(32, 512); self.spnRes.setSingleStep(16); self.spnRes.setValue(128)
        vbox_ctrl.addWidget(self.spnRes)

        vbox_ctrl.addWidget(QLabel("BP smoothing (0=none, 0.9=heavy)"))
        self.spnBPAlpha = QDoubleSpinBox()
        self.spnBPAlpha.setDecimals(2); self.spnBPAlpha.setRange(0.0, 0.95)
        self.spnBPAlpha.setSingleStep(0.05); self.spnBPAlpha.setValue(self.mySolver.bp_temporal_alpha)
        vbox_ctrl.addWidget(self.spnBPAlpha)

        btnApplyLevels = QPushButton("Apply colormap/levels")
        btnRebuildMesh = QPushButton("Rebuild mesh")
        vbox_ctrl.addWidget(btnApplyLevels)
        vbox_ctrl.addWidget(btnRebuildMesh)
        vbox_ctrl.addStretch(1)

        btnApplyLevels.clicked.connect(self._apply_levels_and_cmap)
        btnRebuildMesh.clicked.connect(self._rebuild_mesh_from_controls)
        self.spnBPAlpha.valueChanged.connect(self._update_bp_alpha)

        self.tabs.addTab(tabControls, "Controls")
        self.add_switch_button()

        self._last_t = None
        self._fps_alpha = 0.9
        self._fps_est = 0.0

    # ------------------------------------------------------------------
    # Grid rasterization
    # ------------------------------------------------------------------

    def _build_raster_cache(self):
        """Pre-computes barycentric weights for each grid pixel — once per mesh."""
        pts = self.mySolver.mesh_obj.node
        tri = self.mySolver.mesh_obj.element

        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        self._grid_extent = [x_min, x_max, y_min, y_max]

        xs = np.linspace(x_min, x_max, self._nx)
        ys = np.linspace(y_min, y_max, self._ny)
        grid_x, grid_y = np.meshgrid(xs, ys)

        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
        X, Y = grid_x.ravel(), grid_y.ravel()
        finder = triang.get_trifinder()
        tri_idx = finder(X, Y)

        n_pix = tri_idx.size
        v0 = np.full(n_pix, -1, dtype=int)
        v1 = np.full(n_pix, -1, dtype=int)
        v2 = np.full(n_pix, -1, dtype=int)
        w0 = np.zeros(n_pix)
        w1 = np.zeros(n_pix)
        w2 = np.zeros(n_pix)
        valid = tri_idx >= 0

        if np.any(valid):
            idxs = tri_idx[valid]
            for t in np.unique(idxs):
                pix_ids = np.nonzero(valid)[0][idxs == t]
                T = tri[t]
                A, B, C = pts[T[0]], pts[T[1]], pts[T[2]]
                Px, Py = X[pix_ids], Y[pix_ids]
                den = (B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1])
                if den == 0.0:
                    continue
                l0 = ((B[1]-C[1])*(Px-C[0]) + (C[0]-B[0])*(Py-C[1])) / den
                l1 = ((C[1]-A[1])*(Px-C[0]) + (A[0]-C[0])*(Py-C[1])) / den
                l2 = 1.0 - l0 - l1
                v0[pix_ids]=T[0]; v1[pix_ids]=T[1]; v2[pix_ids]=T[2]
                w0[pix_ids]=l0;   w1[pix_ids]=l1;   w2[pix_ids]=l2

        self._raster_cache = {"v0":v0,"v1":v1,"v2":v2,
                               "w0":w0,"w1":w1,"w2":w2,"mask":valid}

    def _rasterize(self, nodal_vals: np.ndarray) -> np.ndarray:
        """Interpolates nodal values onto pixel grid using cached barycentric weights."""
        c = self._raster_cache
        valid = c["mask"]
        img_flat = np.full(c["v0"].shape, np.nan)
        idx = np.nonzero(valid)[0]
        img_flat[idx] = (c["w0"][idx]*nodal_vals[c["v0"][idx]] +
                         c["w1"][idx]*nodal_vals[c["v1"][idx]] +
                         c["w2"][idx]*nodal_vals[c["v2"][idx]])
        return img_flat.reshape(self._ny, self._nx)

    def _mesh_to_nodal(self, vals: np.ndarray) -> np.ndarray:
        """Converts per-element values to per-node if needed."""
        pts = self.mySolver.mesh_obj.node
        tri = self.mySolver.mesh_obj.element
        v = np.real(vals)
        if v.ndim > 1:
            v = np.ravel(v)
        if v.shape[0] == pts.shape[0]:
            return v
        elif v.shape[0] == tri.shape[0]:
            return sim2pts(pts, tri, v)
        return np.zeros(pts.shape[0])

    def _rasterize_method(self, method: str) -> np.ndarray:
        """Returns (ny, nx) smooth grid image for bp or jac."""
        if method == 'bp':
            # BP: use normalized+smoothed image already in mySolver.image
            nodal = self._mesh_to_nodal(self.mySolver.image)
        else:
            # JAC: convert ds_med_frame to nodal then rasterize
            nodal = self._mesh_to_nodal(self.mySolver.ds_med_frame)
        return self._rasterize(nodal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_bp_alpha(self, value):
        self.mySolver.bp_temporal_alpha = float(value)

    def _on_timer(self):
        self.update_plot(self.data, self.nframes, method=self.method)

    def _attach_or_update_colorbar(self, mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if self._colorbar_ref is None or self._cbar_ax is None:
            divider = make_axes_locatable(self.eitImage.axes)
            self._cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
            self._colorbar_ref = self.eitImage.fig.colorbar(mappable, cax=self._cbar_ax)
        else:
            self._colorbar_ref.update_normal(mappable)

    def _lock_axes_extent(self):
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
        ax.set_aspect('equal', adjustable='box')

    def _apply_levels_and_cmap(self):
        cmap_name = self.cmbCmap.currentText()
        try:
            self._plotImage_ref.set_cmap(cmap_name)
        except Exception:
            print(f"Colormap {cmap_name} not supported.")
        vmin = float(self.spnVmin.value())
        vmax = float(self.spnVmax.value())
        if hasattr(self._plotImage_ref, 'set_clim'):
            self._plotImage_ref.set_clim(vmin, vmax)
        self.eitImage.draw()

    # ------------------------------------------------------------------
    # Solver switching
    # ------------------------------------------------------------------

    def update_solver(self, new_method: str):
        if new_method == self.method:
            return
        try:
            self.timer.stop()
        except Exception:
            pass

        self.method = new_method

        self.eitImage.axes.clear()
        self.eitMeasurementsSE.axes.clear()
        self.eitMeasurementsDiff.axes.clear()

        if self._colorbar_ref is not None:
            try:
                self._colorbar_ref.remove()
            except Exception:
                pass
            self._colorbar_ref = None
            self._cbar_ax = None

        self.mySolver.recreate_mesh(method=self.method, h0=0.1)
        self.mySolver.setVref(self.data[0])

        self._plotImage_ref = None
        self._plotSE_ref = None
        self._plotDiff_ref = None
        self.frameCounter = 0

        # Rebuild raster cache for new mesh
        self._nx = self._ny = int(self.spnRes.value())
        self._raster_cache = None
        self._build_raster_cache()

        self.init_plots(data=self.data, method=self.method)
        self.update_plot(self.data, self.nframes, method=self.method)
        self.timer.start()

    # ------------------------------------------------------------------
    # Plot initialisation
    # ------------------------------------------------------------------

    def init_plots(self, data, method='greit'):
        """Initialises all plot objects. Caller must set Vref first."""
        self.frameCounter = 0
        self.dataSE = data[0]

        # Measurement curves
        self._plotSE_ref = self.eitMeasurementsSE.axes.plot(self.dataSE, lw=1)[0]
        diff0 = self.mySolver.se_to_diff(self.dataSE)
        self._plotDiff_ref = self.eitMeasurementsDiff.axes.plot(diff0, lw=1)[0]

        try:
            se_min, se_max = float(self.data.min()), float(self.data.max())
        except Exception:
            se_min, se_max = float(self.dataSE.min()), float(self.dataSE.max())
        self.eitMeasurementsSE.axes.set_ylim(se_min, se_max)

        dmin, dmax = float(diff0.min()), float(diff0.max())
        pad = 0.05 * (dmax - dmin + 1e-9)
        self.eitMeasurementsDiff.axes.set_ylim(dmin - pad, dmax + pad)

        # Build raster cache on first call
        if self._raster_cache is None:
            self._nx = self._ny = 128
            self._build_raster_cache()

        self._lock_axes_extent()

        # Run solver for frame 0 so mySolver.image is populated
        self.mySolver.updateImage(data[self.frameCounter], method, plot_ref=None)

        if method == 'greit':
            img0 = np.real(self.mySolver.image)
            self._plotImage_ref = self.eitImage.axes.imshow(
                img0, origin='lower', vmin=-1.0, vmax=1.0,
                extent=[self._ax_xlim[0], self._ax_xlim[1],
                        self._ax_ylim[0], self._ax_ylim[1]],
                interpolation='bilinear'
            )
        elif method in ('bp', 'jac'):
            img0 = self._rasterize_method(method)
            self._plotImage_ref = self.eitImage.axes.imshow(
                img0, origin='lower', vmin=-1.0, vmax=1.0,
                extent=[self._grid_extent[0], self._grid_extent[1],
                        self._grid_extent[2], self._grid_extent[3]],
                interpolation='bilinear'
            )

        self._attach_or_update_colorbar(self._plotImage_ref)

        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")
        self.eitMeasurementsSE.axes.set_title(f"Single-Ended Measurements ({self.frameCounter})")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")

        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update_plot(self, data, nframes, method):
        if not self.isVisible():
            return

        self.dataSE = data[self.frameCounter]
        self.dataDiff = self.mySolver.se_to_diff(self.dataSE)

        self._plotSE_ref.set_ydata(self.dataSE)
        self._plotDiff_ref.set_ydata(self.dataDiff)

        # Run solver — normalization + smoothing happen inside updateImage
        self.mySolver.updateImage(self.dataSE, method, plot_ref=None)

        if method == 'greit':
            self._plotImage_ref.set_data(np.real(self.mySolver.image))
        elif method in ('bp', 'jac'):
            img = self._rasterize_method(method)
            self._plotImage_ref.set_data(img)
            self._plotImage_ref.set_clim(-1.0, 1.0)

        self.eitImage.axes.set_xlim(*self._ax_xlim)
        self.eitImage.axes.set_ylim(*self._ax_ylim)

        self.eitImage.axes.set_title(f"Frame {self.frameCounter}")
        self.eitMeasurementsSE.axes.set_title(f"Single-Ended Measurements ({self.frameCounter})")
        self.eitMeasurementsDiff.axes.set_title(f"Differential Measurements ({self.frameCounter})")

        self.eitImage.draw()
        self.eitMeasurementsSE.draw()
        self.eitMeasurementsDiff.draw()

        self.frameCounter = (self.frameCounter + 1) % nframes

        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_est = self._fps_alpha * self._fps_est + (1 - self._fps_alpha) * inst_fps
                try:
                    self.setWindowTitle(
                        f"EITduino — {self.method.upper()}  |  FPS: {self._fps_est:.1f}"
                    )
                except Exception:
                    pass
        self._last_t = t

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def add_switch_button(self):
        btn_switch = QPushButton("Switch to PyQtGraph")
        btn_switch.clicked.connect(self.switch_to_pyqtgraph)
        self.layout_gui.addWidget(btn_switch, 1, 0)

    def switch_to_pyqtgraph(self):
        try:
            self._dispose_matplotlib()
        except Exception:
            pass
        def _launch():
            from pyqtgraph_interface import MainWindowPG
            app = QApplication.instance()
            new_win = MainWindowPG(self.data, self.nframes, method=self.method)
            app.setProperty('active_window', new_win)
            new_win.show()
        QTimer.singleShot(0, _launch)
        self.close()

    def _rebuild_mesh_from_controls(self):
        n_el = self.mySolver.n_el
        res = int(self.spnRes.value())
        h0_trials = [0.1 * (0.8 ** k) for k in range(6)]
        success = False
        last_err = None

        for h0_try in h0_trials:
            try:
                self.mySolver.recreate_mesh(
                    n_el=n_el, fd=self.mySolver.fd, method=self.method, h0=h0_try
                )
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
            if self._colorbar_ref is not None:
                try:
                    self._colorbar_ref.remove()
                except Exception:
                    pass
                self._colorbar_ref = None
                self._cbar_ax = None
            self._plotImage_ref = None
            self._plotSE_ref = None
            self._plotDiff_ref = None
            self.frameCounter = 0
            self.init_plots(data=self.data, method=self.method)
            self.update_plot(self.data, self.nframes, method=self.method)
        else:
            QMessageBox.warning(self, "Rebuild mesh", f"Failed to rebuild mesh.\nDetail: {last_err}")

    def _dispose_matplotlib(self):
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._on_timer)
        except Exception:
            pass
        if self._colorbar_ref is not None:
            try:
                self._colorbar_ref.remove()
            except Exception:
                pass
            self._colorbar_ref = None
            self._cbar_ax = None
        for canvas in [self.eitImage, self.eitMeasurementsSE, self.eitMeasurementsDiff]:
            try:
                canvas.axes.cla(); canvas.draw()
            except Exception:
                pass
            try:
                canvas.setParent(None); canvas.deleteLater()
            except Exception:
                pass
        try:
            plt.close(self.eitImage.fig)
            plt.close(self.eitMeasurementsSE.fig)
            plt.close(self.eitMeasurementsDiff.fig)
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._dispose_matplotlib()
        except Exception:
            pass
        super().closeEvent(event)