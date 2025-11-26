from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import circle, ellipse, rectangle
import pyeit.mesh.shape as shape

# Nota: EITForward importado mas não usado
# from pyeit.eit.fem import EITForward

class EITsolver:
    def __init__(
        self,
        n_el=8,
        fd=shape.circle,
        h0=0.1,
        method='greit',
        parser_meas="rotate_meas",
        lamb=0.01, # Este parâmetro ajuda na suavização, talvez aumentar este valor possa ajudar com o problema da matriz singular
        p=0.5,
    ):
        # buffers de sinais
        self.Vref = np.asarray([])
        self.Vmeas = np.asarray([])
        self.Vse = np.asarray([])

        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0
        self.parser_meas = parser_meas
        self.hp = {"lamb": lamb, "p": p, "bp_weight": "none"}  # ajustes em um só lugar

        # estado do solver
        self._is_ready = False
        self._build_solver()     # cria malha/protocolo/solver
        self.setup()             # define como pronto o solver

    def _build_solver(self):
        """(Re)cria malha, protocolo e instancia o solver da classe escolhida."""
        self.mesh_obj = mesh.create(self.n_el, h0=self.h0, fd=self.fd)
        self.protocol_obj = protocol.create(
            self.n_el, dist_exc=1, step_meas=1, parser_meas=self.parser_meas
        )
        self.__create_vec_se_to_diff__()

        if self.method == "bp":
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
        elif self.method == "greit":
            self.eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
        elif self.method == "jac":
            self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
        else:
            raise Exception(f"Method {self.method} unknown.")

        # acabou de trocar tudo → ainda não está preparado
        self._is_ready = False

    def setup(self):
        """Chama setup do solver com os hiperparâmetros corretos e marca como pronto."""
        p = self.hp["p"]
        lamb = self.hp["lamb"]

        if self.method == "bp":
            # 'unit' é comum; você testou 'none'. Mantemos seu padrão:
            self.eit.setup(weight=self.hp["bp_weight"])
        elif self.method == "greit":
            # GREIT costuma aceitar p, lamb, perm, jac_normalized
            self.eit.setup(p=p, lamb=lamb, perm=1, jac_normalized=True)
        elif self.method == "jac":
            # Ajuste conforme necessidade: method='kotre', p (norma Lp), e lamb (Tikhonov)
            self.eit.setup(p=p, lamb=lamb, method="kotre", perm=1, jac_normalized=True)

        self._is_ready = True

    def ensure_ready(self):
        """Garante que o solver foi 'setup()' antes de operar."""
        if not getattr(self, "_is_ready", False):
            self.setup()

    def __create_vec_se_to_diff__(self):
        vec_a_all = np.array([])
        vec_b_all = np.array([])

        for idx, vec in enumerate(self.protocol_obj.meas_mat[:, :, :]):
            vec_a = (idx * self.n_el) + vec[:, 0]
            vec_b = (idx * self.n_el) + vec[:, 1]
            vec_a_all = np.append(vec_a_all, vec_a)
            vec_b_all = np.append(vec_b_all, vec_b)
        self.vec_a_all = vec_a_all.astype(int)
        self.vec_b_all = vec_b_all.astype(int)

    def se_to_diff(self, v_se):
        return v_se[self.vec_b_all] - v_se[self.vec_a_all]

    def setVref(self, VrefSe):
        """Define Vref (em diferencial) e garante prontidão do solver."""
        self.Vref = self.se_to_diff(VrefSe)
        self.ensure_ready()

   
    def recreate_mesh(
        self, n_el=8, fd=shape.circle, h0=0.1, method='greit', parser_meas="rotate_meas", lamb=0.01, p=0.5,
    ):
        """Recria malha e talvez troca o método. Não chama setup aqui."""
        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0
        self.parser_meas = parser_meas
        self.hp.update({"lamb": lamb, "p": p})

        self._build_solver()  

    def setframes(self, Vse, method=None):
        """Recebe SE, converte para diferencial e resolve. Garante setup antes de solve()."""
        # Se method diferente do atual, troca e invalida.
        if method is not None and method != self.method:
            self.recreate_mesh(
                n_el=self.n_el,
                fd=self.fd,
                h0=self.h0,
                method=method,
                parser_meas=self.parser_meas,
                lamb=self.hp["lamb"],
                p=self.hp["p"],
            )

        self.Vse = Vse
        self.Vmeas = self.se_to_diff(Vse)

        # Se Vref ainda não definido, cai no fallback
        if self.Vref.size == 0:
            # fallback: usa o próprio frame como Vref
            self.Vref = self.Vmeas.copy()

        self.ensure_ready()
        self.ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)

        # cache JAC no espaço de nós (útil em algumas visualizações)
        if self.method == "jac":
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element
            self.ds_n = sim2pts(pts, tri, np.real(self.ds_med_frame))

        return self.ds_med_frame

    def updateImage(self, Vse, method, plot_ref=None):
        """Atualiza self.image de acordo com o método atual."""
        self.setframes(Vse, method)

        if self.method == 'greit':
            x, y, ds_med_frame = self.eit.mask_value(self.ds_med_frame, mask_value=np.nan)
            self.image = np.real(ds_med_frame)
            if plot_ref is not None:
                plot_ref.set_data(self.image)

        elif self.method == 'bp':
            self.image = np.real(self.ds_med_frame)
            if plot_ref is not None:
                plot_ref.set_array(self.image)

        elif self.method == 'jac':
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element
            self.image = sim2pts(pts, tri, self.ds_med_frame)
            if plot_ref is not None:
                plot_ref.set_array(self.image)

        return self.image