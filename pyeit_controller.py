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
    def __init__(self, n_el=8, fd=shape.circle, h0=0.2, method='greit', parser_meas="rotate_meas", lamb=0.1, p=0.5):
        self.Vref=np.asarray([])
        self.Vmeas=np.asarray([])
        self.vec_a_all=np.asarray([])
        self.vec_b_all=np.asarray([])
        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0        
        self.mesh_obj = mesh.create(n_el, h0=h0, fd=fd)
        
        if method == "bp":
            self.protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas)
            self.__create_vec_se_to_diff__()
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
            self.eit.setup(weight="none")

        elif method == "greit":
            self.protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas)
            self.__create_vec_se_to_diff__()
            self.eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=0.50, lamb=0.01, perm=1)

        elif method == "jac":
            self.protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")
            self.__create_vec_se_to_diff__()
            self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=p, lamb=lamb, method="kotre", perm=1)
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

        if self.method=='greit':
            ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)
            x, y ,ds_med_frame = self.eit.mask_value(ds_med_frame, mask_value=np.nan) #para 'greit'
            self.image = np.real(ds_med_frame)
            
            if plot_ref!=None:
                plot_ref.set_data(self.image)
        
        elif self.method=='bp':
            self.image = self.eit.solve(self.Vmeas, self.Vref, normalize=True)

            if plot_ref!=None:
                plot_ref.set_data(self.image)

        elif self.method =='jac':
            ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)
            ds_med_frame_n = sim2pts(self.mesh_obj.node, self.mesh_obj.element, np.real(ds_med_frame))
            # self.image = self.mesh_obj.node[:, 0], self.mesh_obj.node[:, 1], self.mesh_obj.element, ds_med_frame_n
            self.image = ds_med_frame_n

            if plot_ref!=None:
                plot_ref.set_data(self.image)
                
        return self.image