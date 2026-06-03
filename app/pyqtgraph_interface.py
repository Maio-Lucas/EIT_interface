from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QTabWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QCheckBox,
    QHBoxLayout, QSpinBox, QMessageBox, QApplication
)
from PyQt6.QtCore import QTimer
import numpy as np
import pyqtgraph as pg
import pyeit.mesh.shape as shape
import matplotlib.tri as mtri

from pyeit_controller import EITsolver
from pyeit.eit.interp2d import sim2pts
import warnings
import time


class MainWindowPG(QMainWindow):
    def __init__(self, data, nframes, method='greit'):
        super().__init__()
        self.setWindowTitle("EITduino (PyQtGraph)")
        self.data = data
        self.nframes = nframes
        self.method = method if method in ('greit', 'bp', 'jac') else 'greit'
        self.frame = 0

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Measurement plots
        self.plotSE = pg.PlotWidget(title="Single-Ended")
        self.plotSE.showGrid(x=True, y=True, alpha=0.3)
        self.plotDiff = pg.PlotWidget(title="Differential")
        self.plotDiff.showGrid(x=True, y=True, alpha=0.3)
        self.curveSE = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        # Image view
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.imageView.view.setAspectLocked(True)
        self.imageView.getView().invertY(True)
        try:
            self.cmap = pg.colormap.getFromMatplotlib('RdBu_r')
        except Exception:
            self.cmap = pg.colormap.get('CET-D1')
        self.imageView.setColorMap(self.cmap)
        self.imageView.ui.roiBtn.hide()
        self.imageView.ui.menuBtn.hide()
        self.imageView.getImageItem().setAutoDownsample(True)
        self.imageView.setMinimumSize(640, 640)

        # Solver tab
        tabSolver = QWidget()
        vbox_solver = QVBoxLayout(tabSolver)
        vbox_solver.setSpacing(8)
        vbox_solver.addWidget(QLabel("Solver method"))
        btnBP = QPushButton('BP');     btnBP.clicked.connect(lambda: self.update_solver('bp'))
        btnJAC = QPushButton('JAC');   btnJAC.clicked.connect(lambda: self.update_solver('jac'))
        btnGREIT = QPushButton('GREIT'); btnGREIT.clicked.connect(lambda: self.update_solver('greit'))
        vbox_solver.addWidget(btnBP)
        vbox_solver.addWidget(btnJAC)
        vbox_solver.addWidget(btnGREIT)
        vbox_solver.addStretch(1)

        # Measurements tab
        tabMeas = QWidget()
        vbox_meas = QVBoxLayout(tabMeas)
        vbox_meas.addWidget(self.plotSE)
        vbox_meas.addWidget(self.plotDiff)

        self.tabs = QTabWidget()
        self.tabs.addTab(tabSolver, "Solver")
        self.tabs.addTab(tabMeas, "Measurements")
        self.tabs.setMinimumWidth(280)

        # Solver — setVref before anything else
        self.solver = EITsolver(method=self.method, h0=0.1)
        self.solver.setVref(self.data[self.frame])

        self.x_min = self.x_max = 0.0
        self.y_min = self.y_max = 0.0
        self.nx = self.ny = 128
        self._triang = None
        self.pos_xy = (0.0, 0.0)
        self.scale_xy = (1.0, 1.0)
        self._raster_cache = None
        self._cache_key_mesh = None

        self._prepare_grid_and_triangulation()

        # Controls tab
        tabControls = QWidget()
        vbox_ctrl = QVBoxLayout(tabControls)
        vbox_ctrl.setSpacing(8)

        vbox_ctrl.addWidget(QLabel("Colormap"))
        self.cmbCmap = QComboBox()
        self.cmbCmap.addItems(["RdBu_r", "viridis", "CET-D1", "plasma", "inferno"])
        vbox_ctrl.addWidget(self.cmbCmap)

        vbox_ctrl.addWidget(QLabel("Levels (min / max)"))
        row_levels = QHBoxLayout()
        self.spnVmin = QDoubleSpinBox(); self.spnVmin.setDecimals(3)
        self.spnVmin.setRange(-1e6, 1e6); self.spnVmin.setValue(-1.0)
        self.spnVmax = QDoubleSpinBox(); self.spnVmax.setDecimals(3)
        self.spnVmax.setRange(-1e6, 1e6); self.spnVmax.setValue(1.0)
        row_levels.addWidget(self.spnVmin); row_levels.addWidget(self.spnVmax)
        vbox_ctrl.addLayout(row_levels)

        self.chkAutoLevels = QCheckBox("Auto-levels")
        self.chkAutoLevels.setChecked(False)
        vbox_ctrl.addWidget(self.chkAutoLevels)

        vbox_ctrl.addWidget(QLabel("# of electrodes (requires rebuild)"))
        self.spnNel = QSpinBox(); self.spnNel.setRange(8, 32); self.spnNel.setSingleStep(2)
        self.spnNel.setValue(self.solver.n_el); self.spnNel.setEnabled(False)
        vbox_ctrl.addWidget(self.spnNel)

        vbox_ctrl.addWidget(QLabel("Mesh shape"))
        self.cmbShape = QComboBox()
        self.cmbShape.addItems(["circle", "ellipse", "rectangle"])
        vbox_ctrl.addWidget(self.cmbShape)

        vbox_ctrl.addWidget(QLabel("Grid resolution (nx = ny)"))
        self.spnRes = QSpinBox(); self.spnRes.setRange(32, 512)
        self.spnRes.setSingleStep(16); self.spnRes.setValue(self.nx)
        vbox_ctrl.addWidget(self.spnRes)

        vbox_ctrl.addWidget(QLabel("BP smoothing (0=none, 0.9=heavy)"))
        self.spnBPAlpha = QDoubleSpinBox(); self.spnBPAlpha.setDecimals(2)
        self.spnBPAlpha.setRange(0.0, 0.95); self.spnBPAlpha.setSingleStep(0.05)
        self.spnBPAlpha.setValue(self.solver.bp_temporal_alpha)
        vbox_ctrl.addWidget(self.spnBPAlpha)

        btnApplyLevels = QPushButton("Apply colormap/levels")
        btnRebuildMesh = QPushButton("Rebuild mesh")
        vbox_ctrl.addWidget(btnApplyLevels)
        vbox_ctrl.addWidget(btnRebuildMesh)
        vbox_ctrl.addStretch(1)

        btnApplyLevels.clicked.connect(self._apply_levels_and_cmap)
        btnRebuildMesh.clicked.connect(self._rebuild_mesh_from_controls)
        self.spnBPAlpha.valueChanged.connect(lambda v: setattr(self.solver, 'bp_temporal_alpha', float(v)))

        self.tabs.addTab(tabControls, "Controls")

        # Layout
        grid = QGridLayout()
        grid.addWidget(self.tabs,      0, 0, 2, 1)
        grid.addWidget(self.imageView, 0, 1, 2, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 3)
        grid.setRowStretch(0, 1);    grid.setRowStretch(1, 1)
        container = QWidget()
        self._safe_set_layout(container, grid)
        self.setCentralWidget(container)

        self._init_plots()

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update_plots)
        self.timer.start()

        self.add_switch_button()

        self._last_t = None
        self._fps_alpha = 0.9
        self._fps_est = 0.0

    # ------------------------------------------------------------------
    # Grid rasterization (shared with pyqt_interface logic)
    # ------------------------------------------------------------------

    def _suggest_grid_resolution(self, tri_count: int) -> int:
        if tri_count < 500:   return 128
        elif tri_count < 1000: return 160
        else:                  return 192

    def _has_degenerate_elements(self, mesh_obj, eps=1e-14):
        pts = mesh_obj.node; tri = mesh_obj.element
        a=pts[tri[:,0]]; b=pts[tri[:,1]]; c=pts[tri[:,2]]
        areas = 0.5*np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0]))
        return np.any(areas <= eps)

    def _validate_mesh_or_raise(self, mesh_obj):
        pts = getattr(mesh_obj, "node", None)
        tri = getattr(mesh_obj, "element", None)
        if pts is None or tri is None or pts.size == 0 or tri.size == 0:
            raise RuntimeError("Empty mesh.")
        if self._has_degenerate_elements(mesh_obj):
            raise RuntimeError("Degenerate triangles in mesh.")

    def _prepare_grid_and_triangulation(self):
        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element

        self.x_min, self.x_max = float(pts[:,0].min()), float(pts[:,0].max())
        self.y_min, self.y_max = float(pts[:,1].min()), float(pts[:,1].max())

        res = int(self.spnRes.value()) if hasattr(self, 'spnRes') else \
              self._suggest_grid_resolution(tri.shape[0])
        self.nx = self.ny = res

        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        self._triang = mtri.Triangulation(pts[:,0], pts[:,1], tri)
        self.dx = (self.x_max-self.x_min)/(self.nx-1)
        self.dy = (self.y_max-self.y_min)/(self.ny-1)
        self.pos_xy   = (self.x_min, self.y_min)
        self.scale_xy = (self.dx, self.dy)
        self._build_raster_cache()

    def _build_raster_cache(self):
        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element
        X = self.grid_x.ravel(); Y = self.grid_y.ravel()
        finder  = self._triang.get_trifinder()
        tri_idx = finder(X, Y)
        n_pix = tri_idx.size
        v0=np.full(n_pix,-1,dtype=int); v1=v0.copy(); v2=v0.copy()
        w0=np.zeros(n_pix); w1=w0.copy(); w2=w0.copy()
        valid = tri_idx >= 0
        if np.any(valid):
            idxs = tri_idx[valid]
            for t in np.unique(idxs):
                pix_ids = np.nonzero(valid)[0][idxs==t]
                T=tri[t]; A,B,C=pts[T[0]],pts[T[1]],pts[T[2]]
                Px,Py=X[pix_ids],Y[pix_ids]
                den=(B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1])
                if den==0.0: continue
                l0=((B[1]-C[1])*(Px-C[0])+(C[0]-B[0])*(Py-C[1]))/den
                l1=((C[1]-A[1])*(Px-C[0])+(A[0]-C[0])*(Py-C[1]))/den
                l2=1.0-l0-l1
                v0[pix_ids]=T[0]; v1[pix_ids]=T[1]; v2[pix_ids]=T[2]
                w0[pix_ids]=l0;   w1[pix_ids]=l1;   w2[pix_ids]=l2
        self._raster_cache = {"v0":v0,"v1":v1,"v2":v2,
                               "w0":w0,"w1":w1,"w2":w2,"mask":valid}
        self._cache_key_mesh = (self.solver.n_el, self.solver.fd.__name__, self.nx, self.ny)

    def _rasterize_with_cache(self, nodal_vals: np.ndarray) -> np.ndarray:
        c = self._raster_cache
        valid = c["mask"]
        img_flat = np.full(c["v0"].shape, np.nan)
        idx = np.nonzero(valid)[0]
        img_flat[idx] = (c["w0"][idx]*nodal_vals[c["v0"][idx]] +
                         c["w1"][idx]*nodal_vals[c["v1"][idx]] +
                         c["w2"][idx]*nodal_vals[c["v2"][idx]])
        return img_flat.reshape(self.ny, self.nx)

    def _mesh_to_nodal(self, vals: np.ndarray) -> np.ndarray:
        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element
        v = np.real(vals)
        if v.ndim > 1: v = np.ravel(v)
        if v.shape[0] == pts.shape[0]: return v
        elif v.shape[0] == tri.shape[0]: return sim2pts(pts, tri, v)
        return np.zeros(pts.shape[0])

    def _rasterize_mesh_frame(self, method: str, se_vector: np.ndarray) -> np.ndarray:
        """
        Resolves one frame and rasterizes onto a pixel grid.
        BP: uses solver.image (normalized + temporally smoothed).
        JAC: uses solver.ds_med_frame converted to nodal values.
        NaN pixels (outside the domain) are set to 0 for PyQtGraph compatibility.
        """
        self.solver.updateImage(se_vector, method)

        if method == 'bp':
            nodal = self._mesh_to_nodal(self.solver.image)
        else:  # jac
            nodal = self._mesh_to_nodal(self.solver.ds_med_frame)

        img = self._rasterize_with_cache(nodal)
        # PyQtGraph renders NaN as bright artifacts — replace with 0 (background)
        img = np.nan_to_num(img, nan=0.0)
        return img

    # ------------------------------------------------------------------
    # Plot init & update
    # ------------------------------------------------------------------

    def _init_plots(self):
        se0   = self.data[self.frame]
        diff0 = self.solver.se_to_diff(se0)
        self.curveSE.setData(se0.astype(float))
        self.curveDiff.setData(diff0.astype(float))

        try:    se_min, se_max = float(self.data.min()), float(self.data.max())
        except: se_min, se_max = float(se0.min()), float(se0.max())
        self.plotSE.setYRange(se_min, se_max)
        dmin, dmax = float(diff0.min()), float(diff0.max())
        self.plotDiff.setYRange(dmin-0.05*(dmax-dmin+1e-9), dmax+0.05*(dmax-dmin+1e-9))

        if self.method == 'greit':
            self.solver.updateImage(se0, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            h, w = img.shape
            scale_greit = ((self.x_max-self.x_min)/(w-1),
                           (self.y_max-self.y_min)/(h-1))
            self.imageView.setImage(img, autoLevels=False, autoRange=True,
                                    levels=(-1.0, 1.0),
                                    pos=self.pos_xy, scale=scale_greit)
        elif self.method in ('bp', 'jac'):
            img0 = self._rasterize_mesh_frame(self.method, se0)
            self.imageView.setImage(img0, autoLevels=False, autoRange=True,
                                    levels=(-1.0, 1.0),
                                    pos=self.pos_xy, scale=self.scale_xy)
        else:
            self.imageView.setImage(np.zeros((self.ny, self.nx), dtype=float),
                                    autoLevels=True, autoRange=True,
                                    pos=self.pos_xy, scale=self.scale_xy)

        self.imageView.view.setDefaultPadding(0)
        self.imageView.view.setRange(xRange=(self.x_min,self.x_max),
                                     yRange=(self.y_min,self.y_max), padding=0)

    def _update_plots(self):
        if not self.isVisible():
            return

        se   = self.data[self.frame]
        diff = self.solver.se_to_diff(se)
        self.curveSE.setData(se.astype(float))
        self.curveDiff.setData(diff.astype(float))

        vmin = float(self.spnVmin.value())
        vmax = float(self.spnVmax.value())

        if self.method == 'greit':
            self.solver.updateImage(se, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            h, w = img.shape
            scale_greit = ((self.x_max-self.x_min)/(w-1),
                           (self.y_max-self.y_min)/(h-1))
            lvl = None if self.chkAutoLevels.isChecked() else (-1.0, 1.0)
            self.imageView.setImage(img, autoLevels=self.chkAutoLevels.isChecked(),
                                    autoRange=False, levels=lvl,
                                    pos=self.pos_xy, scale=scale_greit)
        elif self.method in ('bp', 'jac'):
            img = self._rasterize_mesh_frame(self.method, se)
            lvl = None if self.chkAutoLevels.isChecked() else (-1.0, 1.0)
            self.imageView.setImage(img, autoLevels=self.chkAutoLevels.isChecked(),
                                    autoRange=False, levels=lvl,
                                    pos=self.pos_xy, scale=self.scale_xy)

        self.frame = (self.frame + 1) % self.nframes

        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_est = self._fps_alpha*self._fps_est + (1-self._fps_alpha)*inst_fps
                try:
                    self.setWindowTitle(
                        f"EITduino (PyQtGraph) — {self.method.upper()}  |  FPS: {self._fps_est:.1f}"
                    )
                except Exception:
                    pass
        self._last_t = t

    # ------------------------------------------------------------------
    # Solver switching
    # ------------------------------------------------------------------

    def update_solver(self, new_method: str):
        if new_method == self.method:
            return
        try: self.timer.stop()
        except Exception: pass

        self.method = new_method
        self.plotSE.clear(); self.plotDiff.clear()
        self.curveSE  = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        self.solver.recreate_mesh(method=self.method, h0=0.1)
        self._validate_mesh_or_raise(self.solver.mesh_obj)
        self.frame = 0
        self.solver.setVref(self.data[self.frame])

        self._prepare_grid_and_triangulation()
        self._init_plots()

        try: self.timer.start()
        except Exception: pass

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _apply_levels_and_cmap(self):
        name = self.cmbCmap.currentText()
        try:    cmap = pg.colormap.getFromMatplotlib(name)
        except: cmap = pg.colormap.get(name)
        self.cmap = cmap
        self.imageView.setColorMap(self.cmap)
        if self.chkAutoLevels.isChecked():
            self.imageView.autoLevels()
        else:
            self.imageView.setLevels(float(self.spnVmin.value()),
                                     float(self.spnVmax.value()))

    def _rebuild_mesh_from_controls(self):
        n_el = int(self.spnNel.value()) if self.spnNel.isEnabled() else self.solver.n_el
        shape_name = self.cmbShape.currentText()
        res = int(self.spnRes.value())
        fd_map = {"circle": shape.circle, "ellipse": shape.ellipse, "rectangle": shape.rectangle}
        fd_primary = fd_map[shape_name]
        h0_base = 0.08 if shape_name == "rectangle" else 0.1
        h0_trials = [h0_base * (0.8**k) for k in range(6)]

        self.timer.stop()
        old_solver = self.solver
        old_nx, old_ny = self.nx, self.ny
        old_cache = self._cache_key_mesh

        try:
            success = False; last_err = None
            for h0_try in h0_trials:
                try:
                    self.solver.recreate_mesh(n_el=n_el, fd=fd_primary,
                                              method=self.method, h0=h0_try)
                    self._validate_mesh_or_raise(self.solver.mesh_obj)
                    self.frame = 0
                    self.solver.setVref(self.data[self.frame])
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        self.solver.setframes(self.data[self.frame], self.method)
                    self.nx = self.ny = res
                    self._prepare_grid_and_triangulation()
                    self._init_plots()
                    success = True; break
                except Exception as e:
                    last_err = e; continue

            if not success and shape_name == "rectangle":
                for h0_try in h0_trials:
                    try:
                        self.solver.recreate_mesh(n_el=n_el, fd=shape.ellipse,
                                                  method=self.method, h0=h0_try)
                        self._validate_mesh_or_raise(self.solver.mesh_obj)
                        self.frame = 0
                        self.solver.setVref(self.data[self.frame])
                        with warnings.catch_warnings():
                            warnings.simplefilter('error')
                            self.solver.setframes(self.data[self.frame], self.method)
                        self.nx = self.ny = res
                        self._prepare_grid_and_triangulation()
                        self._init_plots()
                        success = True; break
                    except Exception as e:
                        last_err = e; continue

            if not success:
                raise RuntimeError(f"Could not rebuild mesh. Last error: {last_err}")

        except Exception as e:
            self.solver = old_solver
            self.nx, self.ny = old_nx, old_ny
            self._cache_key_mesh = old_cache
            self._prepare_grid_and_triangulation()
            self._init_plots()
            QMessageBox.warning(self, "Rebuild mesh",
                                f"Could not rebuild mesh.\n\nDetail: {e}")
        finally:
            self.timer.start()

    # ------------------------------------------------------------------
    # Switch & close
    # ------------------------------------------------------------------

    def add_switch_button(self):
        btn_switch = QPushButton("Switch to PyQt6")
        btn_switch.clicked.connect(self.switch_to_pyqt6)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Controls":
                w = self.tabs.widget(i)
                lay = w.layout() or QVBoxLayout(w)
                lay.addWidget(btn_switch)
                break

    def switch_to_pyqt6(self):
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._update_plots)
        except Exception:
            pass
        def _launch():
            from pyqt_interface import MainWindow
            app = QApplication.instance()
            new_win = MainWindow(self.data, self.nframes, method=self.method)
            app.setProperty('active_window', new_win)
            new_win.show()
        QTimer.singleShot(0, _launch)
        self.close()

    def closeEvent(self, event):
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._update_plots)
        except Exception:
            pass
        for w in [self.plotSE, self.plotDiff, self.imageView, self.tabs]:
            try: w.clear()
            except Exception: pass
            try: w.setParent(None); w.deleteLater()
            except Exception: pass
        super().closeEvent(event)

    def _safe_set_layout(self, widget, layout):
        if widget.layout() is None:
            widget.setLayout(layout)