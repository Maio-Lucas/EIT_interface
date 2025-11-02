# app/pyqtgraph_interface.py
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout
from PyQt6.QtCore import QTimer
import numpy as np
import pyqtgraph as pg

# Interpolação sobre triangulação (somente para cálculo, não renderiza MPL)
import matplotlib.tri as mtri

from pyeit_controller import EITsolver
from pyeit.eit.interp2d import sim2pts  # elemento -> nó


class MainWindowPG(QMainWindow):
    def __init__(self, data, nframes, method='greit'):
        super().__init__()
        self.setWindowTitle("EITduino (PyQtGraph)")
        self.data = data
        self.nframes = nframes
        self.method = method if method in ('greit', 'bp', 'jac') else 'greit'
        self.frame = 0

        # ---- Config visual do PG
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # ---- Curvas SE / Diff
        self.plotSE = pg.PlotWidget(title="Single-Ended")
        self.plotSE.showGrid(x=True, y=True, alpha=0.3)
        self.plotDiff = pg.PlotWidget(title="Differential")
        self.plotDiff.showGrid(x=True, y=True, alpha=0.3)
        self.curveSE = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        # ---- Imagem (usaremos para GREIT e BP rasterizado)
        # Documentação: ImageView + setImage
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.imageView.view.setAspectLocked(True)  # mantém aspecto 1:1
        # Fazemos o sistema de coordenadas ter origem "em baixo" (equivalente ao origin='lower')
        self.imageView.getView().invertY(True)

        # ---- Layout
        grid = QGridLayout()
        grid.addWidget(self.plotSE,    0, 0)
        grid.addWidget(self.plotDiff,  1, 0)
        grid.addWidget(self.imageView, 0, 1, 2, 1)
        container = QWidget()
        container.setLayout(grid)
        self.setCentralWidget(container)

        # ---- Solver (pyEIT)
        self.solver = EITsolver(method=self.method, h0=0.1)
        self.solver.setVref(self.data[self.frame])

        # ---- Preparar malha/grade (para BP/JAC)
        self._prepare_grid_and_triangulation()

        # ---- Primeira renderização
        self._init_plots()

        # ---- Timer único
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # ~20 FPS
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ===================== Preparação de malha/grade =====================
    def _prepare_grid_and_triangulation(self):
        """
        Prepara extents, grade regular e triangulação da malha para rasterizar BP (e JAC futuramente).
        """
        pts = self.solver.mesh_obj.node   # (n_nodes, 2)
        tri = self.solver.mesh_obj.element  # (n_triangles, 3)

        # Limites da malha
        self.x_min, self.x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        self.y_min, self.y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

        # Grade regular (defina a resolução desejada)
        self.nx, self.ny = 128, 128
        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

        # Triangulação (reutilizável entre os frames)
        self._triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

        # Para mapear a imagem nos eixos físicos no ImageView:
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)
        self.pos_xy = (self.x_min, self.y_min)
        self.scale_xy = (self.dx, self.dy)

    # ===================== Inicialização das curvas/imagem =====================
    def _init_plots(self):
        se0 = self.data[self.frame]
        diff0 = self.solver.se_to_diff(se0)

        self.curveSE.setData(se0.astype(float))
        self.curveDiff.setData(diff0.astype(float))

        # Limites estáveis (não reescale a cada frame)
        try:
            se_min, se_max = float(self.data.min()), float(self.data.max())
        except Exception:
            se_min, se_max = float(se0.min()), float(se0.max())
        self.plotSE.setYRange(se_min, se_max)

        dmin, dmax = float(diff0.min()), float(diff0.max())
        pad = 0.05 * (dmax - dmin + 1e-9)
        self.plotDiff.setYRange(dmin - pad, dmax + pad)

        # Imagem inicial por método
        if self.method == 'greit':
            self.solver.updateImage(se0, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            # 1ª chamada com autoRange/autoLevels=True para inicializar
            self.imageView.setImage(img,
                                    autoLevels=True, autoRange=True,
                                    pos=None, scale=None)
        elif self.method == 'bp':
            # Rasteriza o primeiro frame do BP
            img_bp = self._rasterize_bp_frame(se0)
            self.imageView.setImage(img_bp,
                                    autoLevels=True, autoRange=True,
                                    pos=self.pos_xy, scale=self.scale_xy)
        else:
            # Placeholder para outros métodos; podemos rasterizar JAC igual ao BP depois
            self.imageView.setImage(np.zeros((self.ny, self.nx), dtype=float),
                                    autoLevels=True, autoRange=True,
                                    pos=self.pos_xy, scale=self.scale_xy)

    def _rasterize_bp_frame(self, se_vector):
        """
        Calcula a imagem do BP para o frame atual e rasteriza em grade regular (ny, nx).
        Aceita retorno por nó OU por elemento, detectando automaticamente.
        """
        # 1) resolve BP neste frame (atualiza self.solver.ds_med_frame)
        #    use setframes para garantir que a solução do método foi recalculada
        self.solver.setframes(Vse=se_vector, method='bp')

        vals = np.real(self.solver.ds_med_frame)  # vetor vindo do solver
        pts = self.solver.mesh_obj.node           # (n_nodes, 2)
        tri = self.solver.mesh_obj.element        # (n_triangles, 3)

        n_nodes = pts.shape[0]
        n_tris  = tri.shape[0]

        # 2) garanta vetor 1D
        if vals.ndim > 1:
            vals = np.ravel(vals)

        # 3) determine se é nodal ou por elemento
        if vals.shape[0] == n_nodes:
            nodal_vals = vals
        elif vals.shape[0] == n_tris:
            # só converta de elemento->nó se realmente for por elemento
            nodal_vals = sim2pts(pts, tri, vals)
        else:
            raise ValueError(
                f"Formato inesperado do BP: len(vals)={vals.shape[0]} "
                f"(n_nodes={n_nodes}, n_tris={n_tris})"
            )

        # 4) interpola para a grade (LinearTriInterpolator usa valores NODAIS)
        interp = mtri.LinearTriInterpolator(self._triang, nodal_vals)
        zi = interp(self.grid_x, self.grid_y)     # masked array (ny, nx)

        # 5) converte para ndarray e trata NaNs (fora do domínio)
        img = np.array(zi, dtype=float)
        img = np.nan_to_num(img, nan=0.0)

        return img

    # ===================== Atualização por frame =====================
    def _tick(self):
        se = self.data[self.frame]
        diff = self.solver.se_to_diff(se)

        # Curvas
        self.curveSE.setData(se.astype(float))
        self.curveDiff.setData(diff.astype(float))

        # Imagem por método (sem resetar zoom/níveis)
        if self.method == 'greit':
            self.solver.updateImage(se, 'greit')
            img = np.asarray(self.solver.image, dtype=float)
            self.imageView.setImage(img, autoLevels=False, autoRange=False,
                                    pos=None, scale=None)

        elif self.method == 'bp':
            img_bp = self._rasterize_bp_frame(se)
            self.imageView.setImage(img_bp, autoLevels=False, autoRange=False,
                                    pos=self.pos_xy, scale=self.scale_xy)

        # Próximo frame
        self.frame = (self.frame + 1) % self.nframes

    # ===================== Troca de método =====================
    def update_solver(self, new_method: str):
        if new_method == self.method:
            return
        self.timer.stop()
        self.method = new_method

        # Recria curvas
        self.plotSE.clear()
        self.plotDiff.clear()
        self.curveSE = self.plotSE.plot(pen=pg.mkPen('#1f77b4', width=1.2))
        self.curveDiff = self.plotDiff.plot(pen=pg.mkPen('#ff7f0e', width=1.2))

        # Reconfigura solver e refs
        self.solver.recreate_mesh(method=self.method, h0=0.1)
        self.frame = 0
        self.solver.setVref(self.data[self.frame])

        # Como a malha pode mudar (método, parser etc.), refaça grade/triangulação
        self._prepare_grid_and_triangulation()

        # Reinit e segue
        self._init_plots()
        self.timer.start()