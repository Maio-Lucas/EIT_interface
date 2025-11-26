from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QTabWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QCheckBox,
    QHBoxLayout, QSpinBox, QMessageBox, QApplication
)
from PyQt6.QtCore import QTimer
import numpy as np
import pyqtgraph as pg
import pyeit.mesh.shape as shape
import matplotlib.tri as mtri  # apenas para trifinder/triangulação (cálculo)

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

        #  Visual PG 
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        #  Plots Medições 
        self.plotSE = pg.PlotWidget(title="Single-Ended")
        self.plotSE.showGrid(x=True, y=True, alpha=0.3)
        self.plotDiff = pg.PlotWidget(title="Differential")
        self.plotDiff.showGrid(x=True, y=True, alpha=0.3)
        self.curveSE = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        #  Imagem 
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

        # Solver
        tabSolver = QWidget()
        vbox_solver = QVBoxLayout(tabSolver)
        vbox_solver.setSpacing(8)
        vbox_solver.addWidget(QLabel("Solver method"))
        btnBP = QPushButton('BP');   btnBP.clicked.connect(lambda: self.update_solver('bp'))
        btnJAC = QPushButton('JAC'); btnJAC.clicked.connect(lambda: self.update_solver('jac'))
        btnGREIT = QPushButton('GREIT'); btnGREIT.clicked.connect(lambda: self.update_solver('greit'))
        vbox_solver.addWidget(btnBP)
        vbox_solver.addWidget(btnJAC)
        vbox_solver.addWidget(btnGREIT)
        vbox_solver.addStretch(1)

        #Measurements 
        tabMeas = QWidget()
        vbox_meas = QVBoxLayout(tabMeas)
        vbox_meas.setSpacing(8)
        vbox_meas.addWidget(self.plotSE)
        vbox_meas.addWidget(self.plotDiff)

        # Tabs container (esquerda)
        self.tabs = QTabWidget()
        self.tabs.addTab(tabSolver, "Solver")
        self.tabs.addTab(tabMeas,   "Measurements")
        self.tabs.setMinimumWidth(280)

        # Solver e Malha/Grade
        self.solver = EITsolver(method=self.method, h0=0.1)
        self.solver.setVref(self.data[self.frame])

        # (serão definidos em _prepare_grid_and_triangulation)
        self.x_min = self.x_max = 0.0
        self.y_min = self.y_max = 0.0
        self.nx = self.ny = 128
        self.grid_x = self.grid_y = None
        self._triang = None
        self.pos_xy = (0.0, 0.0)
        self.scale_xy = (1.0, 1.0)

        # cache do “interpolador” (barycêntrico)
        self._raster_cache = None       # dict com v0,v1,v2,w0,w1,w2,mask (flatten)
        self._cache_key_mesh = None     # assinatura para invalidar cache quando malha/grade mudam

        self._prepare_grid_and_triangulation()   # também constrói cache raster

        # Abas: Controls 
        tabControls = QWidget()
        vbox_ctrl = QVBoxLayout(tabControls)
        vbox_ctrl.setSpacing(8)

        vbox_ctrl.addWidget(QLabel("Colormap"))
        self.cmbCmap = QComboBox()
        self.cmbCmap.addItems(["RdBu_r", "viridis", "CET-D1", "plasma", "inferno"])
        vbox_ctrl.addWidget(self.cmbCmap)

        vbox_ctrl.addWidget(QLabel("Levels (min / max)"))
        row_levels = QHBoxLayout()
        self.spnVmin = QDoubleSpinBox(); self.spnVmin.setDecimals(3); self.spnVmin.setRange(-1e6, 1e6); self.spnVmin.setValue(-0.75)
        self.spnVmax = QDoubleSpinBox(); self.spnVmax.setDecimals(3); self.spnVmax.setRange(-1e6, 1e6); self.spnVmax.setValue(+0.75)
        row_levels.addWidget(self.spnVmin); row_levels.addWidget(self.spnVmax)
        vbox_ctrl.addLayout(row_levels)

        self.chkAutoLevels = QCheckBox("Auto-levels (override levels on update)")
        self.chkAutoLevels.setChecked(False)
        vbox_ctrl.addWidget(self.chkAutoLevels)

        vbox_ctrl.addWidget(QLabel("# of electrodes (requires rebuild)"))
        self.spnNel = QSpinBox(); self.spnNel.setRange(8, 32); self.spnNel.setSingleStep(2)
        self.spnNel.setValue(self.solver.n_el)
        self.spnNel.setEnabled(False)  # dados gravados → manter n_el
        vbox_ctrl.addWidget(self.spnNel)

        vbox_ctrl.addWidget(QLabel("Mesh shape"))
        self.cmbShape = QComboBox()
        self.cmbShape.addItems(["circle", "ellipse", "rectangle"])
        vbox_ctrl.addWidget(self.cmbShape)

        vbox_ctrl.addWidget(QLabel("Grid resolution (nx = ny)"))
        self.spnRes = QSpinBox(); self.spnRes.setRange(32, 512); self.spnRes.setSingleStep(16)
        self.spnRes.setValue(self.nx)
        vbox_ctrl.addWidget(self.spnRes)

        btnApplyLevels = QPushButton("Apply colormap/levels")
        btnRebuildMesh = QPushButton("Rebuild mesh")
        vbox_ctrl.addWidget(btnApplyLevels)
        vbox_ctrl.addWidget(btnRebuildMesh)
        vbox_ctrl.addStretch(1)

        btnApplyLevels.clicked.connect(self._apply_levels_and_cmap)
        btnRebuildMesh.clicked.connect(self._rebuild_mesh_from_controls)

        self.tabs.addTab(tabControls, "Controls")

        # Layout
        grid = QGridLayout()
        grid.addWidget(self.tabs,      0, 0, 2, 1)
        grid.addWidget(self.imageView, 0, 1, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        container = QWidget()
        self._safe_set_layout(container, grid)
        self.setCentralWidget(container)

        # Primeira renderização e Timer
        self._init_plots()

        self.timer = QTimer(self)
        self.timer.setInterval(50)  # ~20 FPS
        self.timer.timeout.connect(self._update_plots)
        self.timer.start()

        self.add_switch_button()

        self._last_t = None         # timestamp do último frame
        self._fps_alpha = 0.9       # fator de suavização (0.0 = instantâneo; 0.9 = suave)
        self._fps_est = 0.0         # FPS estimado  

    def _suggest_grid_resolution(self, tri_count: int) -> int:
        """Sugere nx=ny adaptativo conforme nº de triângulos (clamp 96..256)."""
        if tri_count < 500:
            res = 128
        elif tri_count < 1000:
            res = 160
        else:
            res = 192
        return int(np.clip(res, 96, 256))
    
    def _has_degenerate_elements(self, mesh_obj, eps=1e-14):
        """Retorna True se houver triângulos com área ~ 0 (degenerados)."""
        pts = mesh_obj.node
        tri = mesh_obj.element
        a = pts[tri[:, 0]]
        b = pts[tri[:, 1]]
        c = pts[tri[:, 2]]
        # área de cada triângulo = 0.5 * | cross(b-a, c-a) |
        areas = 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0]))
        return np.any(areas <= eps)

    def _validate_mesh_or_raise(self, mesh_obj):
        pts = getattr(mesh_obj, "node", None)
        tri = getattr(mesh_obj, "element", None)
        if pts is None or tri is None or pts.size == 0 or tri.size == 0:
            raise RuntimeError("Mesh vazia (sem nós/elementos).")
        if self._has_degenerate_elements(mesh_obj):
            raise RuntimeError("Malha com triângulos degenerados (área ~ 0).")

    def _prepare_grid_and_triangulation(self):
        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element

        self.x_min, self.x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        self.y_min, self.y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

        # resolução adaptativa sugerida (respeita spin se já existir)
        if hasattr(self, "spnRes"):
            res = int(self.spnRes.value())
        else:
            res = self._suggest_grid_resolution(tri.shape[0])

        self.nx = self.ny = res
        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

        self._triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)
        self.pos_xy = (self.x_min, self.y_min)
        self.scale_xy = (self.dx, self.dy)

        # cache de rasterização
        self._build_raster_cache()

    def _build_raster_cache(self):
        """Pré-calcula, para cada pixel da grade, o triângulo e os pesos barycêntricos."""
        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element
        triang = self._triang

        X = self.grid_x.ravel()
        Y = self.grid_y.ravel()
        finder = triang.get_trifinder()
        tri_idx = finder(X, Y)   # -1 fora do domínio

        n_pix = tri_idx.size
        v0 = np.full(n_pix, -1, dtype=int)
        v1 = np.full(n_pix, -1, dtype=int)
        v2 = np.full(n_pix, -1, dtype=int)
        w0 = np.zeros(n_pix, dtype=float)
        w1 = np.zeros(n_pix, dtype=float)
        w2 = np.zeros(n_pix, dtype=float)

        valid = tri_idx >= 0
        if np.any(valid):
            idxs = tri_idx[valid]
            # processa por triângulo para vectorizar
            unique_tris = np.unique(idxs)
            for t in unique_tris:
                mask_t = (idxs == t)
                pix_ids = np.nonzero(valid)[0][mask_t]
                T = tri[t]  # [i,j,k] índices dos nós
                A = pts[T[0]]; B = pts[T[1]]; C = pts[T[2]]

                # coordenadas dos pontos pertencentes a este triângulo
                Px = X[pix_ids]; Py = Y[pix_ids]

                # pesos barycêntricos (método da área)
                den = ((B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1]))
                # evita divisão por zero em triângulos
                if den == 0.0:
                    continue

                l0 = ((B[1] - C[1])*(Px - C[0]) + (C[0] - B[0])*(Py - C[1])) / den
                l1 = ((C[1] - A[1])*(Px - C[0]) + (A[0] - C[0])*(Py - C[1])) / den
                l2 = 1.0 - l0 - l1

                # grava vértices e pesos por pixel
                v0[pix_ids] = T[0]; v1[pix_ids] = T[1]; v2[pix_ids] = T[2]
                w0[pix_ids] = l0;   w1[pix_ids] = l1;   w2[pix_ids] = l2

        self._raster_cache = {
            "v0": v0, "v1": v1, "v2": v2,
            "w0": w0, "w1": w1, "w2": w2,
            "mask": valid
        }
        # chave para invalidar cache quando malha/grade mudarem
        self._cache_key_mesh = (self.solver.n_el, self.solver.fd.__name__, self.nx, self.ny)

    def _rasterize_with_cache(self, nodal_vals: np.ndarray) -> np.ndarray:
        """Usa cache barycêntrico para gerar imagem (ny, nx) a partir de valores nos nós."""
        cache = self._raster_cache
        if cache is None:
            # caso fallback: reconstrói cache e segue
            self._build_raster_cache()
            cache = self._raster_cache

        v0 = cache["v0"]; v1 = cache["v1"]; v2 = cache["v2"]
        w0 = cache["w0"]; w1 = cache["w1"]; w2 = cache["w2"]
        valid = cache["mask"]

        img_flat = np.full(v0.shape, np.nan, dtype=float)
        # calcula apenas para os pontos válidos
        idx_valid = np.nonzero(valid)[0]
        img_flat[idx_valid] = (
            w0[idx_valid] * nodal_vals[v0[idx_valid]] +
            w1[idx_valid] * nodal_vals[v1[idx_valid]] +
            w2[idx_valid] * nodal_vals[v2[idx_valid]]
        )
        return img_flat.reshape(self.ny, self.nx)

    def _init_plots(self):
        se0 = self.data[self.frame]
        diff0 = self.solver.se_to_diff(se0)

        # curvas
        self.curveSE.setData(se0.astype(float))
        self.curveDiff.setData(diff0.astype(float))
        try:
            se_min, se_max = float(self.data.min()), float(self.data.max())
        except Exception:
            se_min, se_max = float(se0.min()), float(se0.max())
        self.plotSE.setYRange(se_min, se_max)

        dmin, dmax = float(diff0.min()), float(diff0.max())
        pad = 0.05 * (dmax - dmin + 1e-9)
        self.plotDiff.setYRange(dmin - pad, dmax + pad)

        # imagem
        if self.method == 'greit':
            self.solver.updateImage(se0, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            h, w = img.shape
            scale_greit = (
                (self.x_max - self.x_min) / (w - 1),
                (self.y_max - self.y_min) / (h - 1)
            )
            self.imageView.setImage(
                img,
                autoLevels=False, autoRange=True,
                levels=(-0.75, 0.75),
                pos=self.pos_xy, scale=scale_greit
            )
        elif self.method in ('bp', 'jac'):
            img0 = self._rasterize_mesh_frame(self.method, se0)
            self.imageView.setImage(
                img0,
                autoLevels=False, autoRange=True,
                levels=(-0.75, 0.75),
                pos=self.pos_xy, scale=self.scale_xy
            )
        else:
            self.imageView.setImage(
                np.zeros((self.ny, self.nx), dtype=float),
                autoLevels=True, autoRange=True,
                pos=self.pos_xy, scale=self.scale_xy
            )

        # enquadra uma vez
        self.imageView.view.setDefaultPadding(0)
        self.imageView.view.setRange(
            xRange=(self.x_min, self.x_max),
            yRange=(self.y_min, self.y_max),
            padding=0
        )
     

    def _rasterize_mesh_frame(self, method: str, se_vector: np.ndarray) -> np.ndarray:
        """BP/JAC: resolve, garante valores nos nós e rasteriza (cache barycêntrico)."""
        self.solver.setframes(Vse=se_vector, method=method)
        vals = np.real(self.solver.ds_med_frame)
        if vals.ndim > 1:
            vals = np.ravel(vals)

        pts = self.solver.mesh_obj.node
        tri = self.solver.mesh_obj.element
        n_nodes = pts.shape[0]
        n_tris  = tri.shape[0]

        if vals.shape[0] == n_nodes:
            nodal = vals
        elif vals.shape[0] == n_tris:
            nodal = sim2pts(pts, tri, vals)
        else:
            raise ValueError(f"Formato inesperado ({vals.shape[0]}); nós={n_nodes}, tris={n_tris}")

        # rasterização rápida via cache
        return self._rasterize_with_cache(nodal)

    def _update_plots(self):
        
        if not self.isVisible():
                return

        se = self.data[self.frame]
        diff = self.solver.se_to_diff(se)

        self.curveSE.setData(se.astype(float))
        self.curveDiff.setData(diff.astype(float))

        if self.method == 'greit':
            self.solver.updateImage(se, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            h, w = img.shape
            scale_greit = (
                (self.x_max - self.x_min) / (w - 1),
                (self.y_max - self.y_min) / (h - 1)
            )
            self.imageView.setImage(
                img,
                autoLevels=False, autoRange=False,
                levels=(-0.75, 0.75),
                pos=self.pos_xy, scale=scale_greit
            )
        elif self.method in ('bp', 'jac'):
            img = self._rasterize_mesh_frame(self.method, se)
            self.imageView.setImage(
                img,
                autoLevels=False, autoRange=False,
                levels=(-0.75, 0.75),
                pos=self.pos_xy, scale=self.scale_xy
            )

        self.frame = (self.frame + 1) % self.nframes

        
        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst_fps = 1.0 / dt
                # Exponential Moving Average (EMA)
                self._fps_est = self._fps_alpha * self._fps_est + (1.0 - self._fps_alpha) * inst_fps
                try:
                    self.setWindowTitle(f"EITduino (PyQtGraph) — {self.method.upper()}  |  FPS: {self._fps_est:.1f}")
                except Exception:
                    pass
        self._last_t = t

    def update_solver(self, new_method: str):
        
        if new_method == self.method:
                return
        try:
            self.timer.stop()
        except Exception:
            pass

        self.method = new_method

        # limpa e recria curvas
        self.plotSE.clear()
        self.plotDiff.clear()
        self.curveSE = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        # recria solver com método novo
        self.solver.recreate_mesh(method=self.method, h0=0.1)
        self._validate_mesh_or_raise(self.solver.mesh_obj)
        self.frame = 0
        self.solver.setVref(self.data[self.frame])

        # refaz malha
        self._prepare_grid_and_triangulation()
        self._init_plots()
        self.timer.start()
        
        try:
            self.timer.start()
        except Exception:
            pass


    def _apply_levels_and_cmap(self):
        # colormap
        name = self.cmbCmap.currentText()
        try:
            cmap = pg.colormap.getFromMatplotlib(name)
        except Exception:
            cmap = pg.colormap.get(name)
        self.cmap = cmap
        self.imageView.setColorMap(self.cmap)

        # levels
        if self.chkAutoLevels.isChecked():
            self.imageView.autoLevels()
        else:
            vmin = float(self.spnVmin.value())
            vmax = float(self.spnVmax.value())
            self.imageView.setLevels(vmin, vmax)

    def _rebuild_mesh_from_controls(self):
        """Reconstrói a malha com tentativas robustas: h0 decrescente e fallback geométrico."""
        n_el = int(self.spnNel.value())
        shape_name = self.cmbShape.currentText()
        res = int(self.spnRes.value())

        # Sempre inicialize fd_soft para evitar UnboundLocalError
        fd_soft = None

        # Seleção do fd nativo (sem kwargs, compatível com sua versão do PyEIT)
        if shape_name == "rectangle":
            fd_primary = shape.rectangle
        else:
            fd_primary = {"circle": shape.circle, "ellipse": shape.ellipse}[shape_name]

        self.timer.stop()
        old_solver = self.solver
        old_nx, old_ny = self.nx, self.ny
        old_cache_key = self._cache_key_mesh

        try:
            if not self.spnNel.isEnabled():
                n_el = self.solver.n_el

            # Conjunto de tentativas (h0 vai diminuindo)
            h0_base = 0.08 if shape_name == "rectangle" else 0.1
            h0_trials = [h0_base * (0.8 ** k) for k in range(6)]  # 6 tentativas

            success = False
            last_err = None

            # 1) Tentativas com o fd primário
            for h0_try in h0_trials:
                try:
                    # Recria solver/malha
                    self.solver.recreate_mesh(n_el=n_el, fd=fd_primary, method=self.method, h0=h0_try)

                    # valida malha criada
                    self._validate_mesh_or_raise(self.solver.mesh_obj)

                    # prepara sinais
                    self.frame = 0
                    self.solver.setVref(self.data[self.frame])

                    # teste precoce de solve com warnings→erro
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        self.solver.setframes(self.data[self.frame], self.method)

                    # se chegou aqui, finalize a troca
                    self.nx = self.ny = res
                    self._prepare_grid_and_triangulation()
                    self._init_plots()
                    success = True
                    break

                except Exception as e:
                    last_err = e
                    print(f"[Rebuild] Falha: shape={shape_name}, h0={h0_try} → {e}")
                    continue

            # 2) Fallback geométrico se era retângulo e nada deu certo
            if not success and shape_name == "rectangle":
                print("[Rebuild] Fallback para elipse (mais estável).")
                for h0_try in h0_trials:
                    try:
                        self.solver.recreate_mesh(n_el=n_el, fd=shape.ellipse, method=self.method, h0=h0_try)
                        self._validate_mesh_or_raise(self.solver.mesh_obj)

                        self.frame = 0
                        self.solver.setVref(self.data[self.frame])

                        with warnings.catch_warnings():
                            warnings.simplefilter('error')
                            self.solver.setframes(self.data[self.frame], self.method)

                        self.nx = self.ny = res
                        self._prepare_grid_and_triangulation()
                        self._init_plots()
                        success = True
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[Rebuild] Fallback elipse falhou com h0={h0_try}: {e}")
                        continue

            if not success:
                raise RuntimeError(f"Falha ao montar malha (singularidade). Último erro: {last_err}")

        except Exception as e:
            # rollback
            self.solver = old_solver
            self.nx, self.ny = old_nx, old_ny
            self._cache_key_mesh = old_cache_key
            self._prepare_grid_and_triangulation()
            self._init_plots()

            QMessageBox.warning(
                self,
                "Rebuild mesh",
                "Não foi possível reconstruir a malha com os parâmetros escolhidos.\n\n"
                f"Detalhe: {e}"
            )
        finally:
            self.timer.start()


    def add_switch_button(self):
        
        btn_switch = QPushButton("Switch to PyQt6")
        btn_switch.clicked.connect(self.switch_to_pyqt6)

        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Controls":
                w = self.tabs.widget(i)
                lay = w.layout()
                if lay is None:
                    lay = QVBoxLayout(w)  # se não achar layout
                lay.addWidget(btn_switch)
                break


    
    def switch_to_pyqt6(self):
        """Fecha esta janela e abre a interface PyQt6 de forma segura."""
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._update_plots)
        except Exception:
            pass

        from PyQt6.QtCore import QTimer
        def _launch():
            from pyqt_interface import MainWindow
            app = QApplication.instance()
            new_win = MainWindow(self.data, self.nframes, method='bp')
            app.setProperty('active_window', new_win)
            new_win.show()
        QTimer.singleShot(0, _launch)

        self.close()

    
    def closeEvent(self, event):
        """Para timer e limpa widgets para evitar AxisItem deleted."""
        try:
            self.timer.stop()
            self.timer.timeout.disconnect(self._update_plots)
        except Exception:
            pass

        # Limpa widgets de gráfico
        for w in [self.plotSE, self.plotDiff, self.imageView, self.tabs]:
            try:
                w.clear()
            except Exception:
                pass
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass

        super().closeEvent(event)


    def _safe_set_layout(self, widget, layout):
        if widget.layout() is None:
            widget.setLayout(layout)
        else:
            pass
