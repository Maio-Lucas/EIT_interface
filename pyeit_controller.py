from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.mesh.shape as shape
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

#Investigar como alterar a posição dos eletrodos
class EITsolver:
    def __init__(self, n_el=16, fd=shape.circle, h0=0.1, method='bp', parser_meas="rotate_meas", lamb=0.01, p=0.5):
        self.Vref=np.asarray([])
        self.Vmeas=np.asarray([])
        self.vec_a_all=np.asarray([])
        self.vec_b_all=np.asarray([])
        self.method = method
        self.n_el = n_el
        self,fd = fd
        self.h0 = h0        
        self.mesh_obj = mesh.create(n_el, h0=h0, fd=fd)
        self.protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas)
        self.__create_vec_se_to_diff__()

        if method == "bp":
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
            self.eit.setup(weight="none")

        elif method == "greit":
            eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
            eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)

        elif method == "jac":
            eit = jac.JAC(self.mesh_obj, self.protocol_obj)
            eit.setup(p=p, lamb=lamb, method="kotre", perm=1, jac_normalized=True)
        else:
            raise Exception(f'Method {method} unknown.')

        def __create_vec_se_to_diff__(self):
            vec_a_all = np.array([])
            vec_b_all = np.array([])
            for idx,vec in enumerate(self.protocol_obj.meas_mat[:,:,:]):
                vec_a = (idx*self.n_el)+vec[:,0]
                vec_b = (idx*self.n_el)+vec[:,1]
                vec_a_all = np.append(vec_a_all,vec_a)
                vec_b_all = np.append(vec_b_all,vec_b)
            self.vec_a_all = vec_a_all.astype(int)
            self.vec_b_all = vec_b_all.astype(int)
        
        def se_to_diff(self, v_se):
            v_diff = v_se[self.vec_b_all] - v_se[self.vec_a_all]
            return v_diff
        
        def setVref(self, VrefSe):
            self.Vref = self.se_to_diff(VrefSe)
        
        def updateImage(self, Vse, plot_ref=None):
            self.Vmeas = self.se_to_diff(Vse)
            ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)
            if self.method=='greit':
                x, y, ds_med_frame = self.eit.mask_value(ds_med_frame, mask_value=np.nan) #para 'greit'
                self.image = np.real(ds_med_frame)
                #print(f'imagem {self.image[15]}')
                if plot_ref!=None:
                    plot_ref.set_data(self.image)
            # IMPLEMENTAR MÉTODOS BP E JAC
            elif self.method=='bp':
                self.image = np.real(ds_med_frame)

            elif self.method =='jac':

                pass

            return self.image

""" 3. naive inverse solver """
solver = []
if solver == "BP":
    eit = bp.BP(mesh_obj, protocol_obj)
    eit.setup(weight="none")
    ds = 192.0 * eit.solve(v1, v0, normalize=True)
elif solver == "GREIT":
    eit = greit.GREIT(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN)
elif solver == "JAC":
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    eit = jac.JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    ds_n = sim2pts(pts, tri, np.real(ds))

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element

# draw
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
# original
ax = axes[0]
ax.axis("equal")
ax.set_title(r"Input $\Delta$ Conductivities")
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
# reconstructed
ax1 = axes[1]
im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds)
ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
ax1.axis("equal")
fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_bp.png', dpi=96)
plt.show()