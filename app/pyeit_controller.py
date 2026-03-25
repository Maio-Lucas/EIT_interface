from __future__ import absolute_import, division, print_function

import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
import pyeit.mesh.shape as shape


class EITsolver:
    def __init__(
        self,
        n_el=8,
        fd=shape.circle,
        h0=0.1,
        method='greit',
        parser_meas="rotate_meas",
        lamb=0.01,
        p=0.5,
        bp_temporal_alpha=0.4,  # temporal smoothing for BP (0=no smoothing, 1=freeze)
    ):
        self.Vref = np.asarray([])
        self.Vmeas = np.asarray([])
        self.Vse = np.asarray([])

        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0
        self.parser_meas = parser_meas
        self.hp = {"lamb": lamb, "p": p, "bp_weight": "std"}

        # Temporal smoothing for BP: image_out = alpha * prev + (1-alpha) * current
        self.bp_temporal_alpha = bp_temporal_alpha
        self._bp_image_prev = None  # stores previous normalized BP image

        self._is_ready = False
        self._build_solver()
        self.setup()

    def _build_solver(self):
        """(Re)creates mesh, protocol and solver instance."""
        self.mesh_obj = mesh.create(self.n_el, h0=self.h0, fd=self.fd)
        self.protocol_obj = protocol.create(
            self.n_el, dist_exc=1, step_meas=1, parser_meas=self.parser_meas
        )
        self.__create_vec_se_to_diff__()

        # IMPORTANT: always reset Vref when mesh/protocol changes
        self.Vref = np.asarray([])
        # Also reset BP smoothing buffer since mesh changed
        self._bp_image_prev = None

        if self.method == "bp":
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
        elif self.method == "greit":
            self.eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
        elif self.method == "jac":
            self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
        else:
            raise Exception(f"Method {self.method} unknown.")

        self._is_ready = False

    def setup(self):
        """Calls solver setup with correct hyperparameters."""
        p = self.hp["p"]
        lamb = self.hp["lamb"]

        if self.method == "bp":
            self.eit.setup(weight=self.hp["bp_weight"])
        elif self.method == "greit":
            self.eit.setup(p=p, lamb=lamb, perm=1, jac_normalized=True)
        elif self.method == "jac":
            self.eit.setup(p=p, lamb=lamb, method="kotre", perm=1, jac_normalized=True)

        self._is_ready = True

    def ensure_ready(self):
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
        """Sets Vref from single-ended data. Always call this before setframes."""
        self.Vref = self.se_to_diff(VrefSe)
        # Reset BP smoothing buffer when reference changes
        self._bp_image_prev = None
        self.ensure_ready()

    def recreate_mesh(
        self, n_el=8, fd=shape.circle, h0=0.1, method='greit',
        parser_meas="rotate_meas", lamb=0.01, p=0.5,
    ):
        """Recreates mesh and possibly changes method. Does NOT call setup."""
        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0
        self.parser_meas = parser_meas
        self.hp.update({"lamb": lamb, "p": p})
        self._build_solver()

    def setframes(self, Vse, method=None):
        """Receives SE data, converts to differential and solves."""
        if method is not None and method != self.method:
            self.recreate_mesh(
                n_el=self.n_el, fd=self.fd, h0=self.h0,
                method=method, parser_meas=self.parser_meas,
                lamb=self.hp["lamb"], p=self.hp["p"],
            )

        self.Vse = Vse
        self.Vmeas = self.se_to_diff(Vse)

        if self.Vref.size == 0:
            # Fallback: use this frame as reference (produces zero image)
            self.Vref = self.Vmeas.copy()

        self.ensure_ready()

        if self.method == "bp":
            self.ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=False)

            diff = self.Vmeas - self.Vref
            print(f"[BP] Vref size: {self.Vref.size} | "
                f"diff norm: {np.linalg.norm(diff):.4f} | "
                f"Vmeas[:3]: {self.Vmeas[:3].round(2)} | "
                f"Vref[:3]: {self.Vref[:3].round(2)}")
        else:
            self.ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)

        if self.method == "jac":
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element
            self.ds_n = sim2pts(pts, tri, np.real(self.ds_med_frame))

        return self.ds_med_frame

    def updateImage(self, Vse, method, plot_ref=None):
        """Updates self.image according to current method."""
        self.setframes(Vse, method)

        if self.method == 'greit':
            x, y, ds_med_frame = self.eit.mask_value(self.ds_med_frame, mask_value=np.nan)
            self.image = np.real(ds_med_frame)
            if plot_ref is not None:
                plot_ref.set_data(self.image)

        elif self.method == 'bp':
            raw = np.real(self.ds_med_frame)

            # Normalize to [-1, 1] using max absolute value
            abs_max = np.abs(raw).max()
            if abs_max > 1e-9:
                normalized = raw / abs_max
            else:
                # Zero frame (e.g. Vref == Vmeas at startup) — use zeros
                normalized = np.zeros_like(raw)

            # Temporal smoothing: reduces frame-to-frame flicker
            # image = alpha * previous + (1 - alpha) * current
            if self._bp_image_prev is None or self._bp_image_prev.shape != normalized.shape:
                self._bp_image_prev = normalized.copy()

            self.image = (
                self.bp_temporal_alpha * self._bp_image_prev +
                (1.0 - self.bp_temporal_alpha) * normalized
            )
            self._bp_image_prev = self.image.copy()

            print(f"[BP] raw min/max: {raw.min():.3f}/{raw.max():.3f} | "
                f"normalized min/max: {normalized.min():.3f}/{normalized.max():.3f} | "
                f"image min/max: {self.image.min():.3f}/{self.image.max():.3f}")

            if plot_ref is not None:
                plot_ref.set_array(self.image)

        elif self.method == 'jac':
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element
            self.image = sim2pts(pts, tri, self.ds_med_frame)
            if plot_ref is not None:
                plot_ref.set_array(self.image)

        return self.image