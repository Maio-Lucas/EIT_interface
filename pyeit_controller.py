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
    def __init__(self, n_el=8, fd=shape.circle, h0=0.1, method='greit', parser_meas="rotate_meas", lamb=0.01, p=0.5):
        self.Vref=np.asarray([])
        self.Vmeas=np.asarray([])
        self.vec_a_all=np.asarray([])
        self.vec_b_all=np.asarray([])
        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0        
        self.mesh_obj = mesh.create(n_el, h0=h0, fd=fd)
        
        self.protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas)
        self.__create_vec_se_to_diff__()

        if method == "bp":
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
            self.eit.setup(weight="none")

        elif method == "greit":
            self.eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=p, lamb=lamb, perm=1, jac_normalized=True)

        elif method == "jac":
            self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=p, lamb=lamb, method="kotre", perm=1, jac_normalized=True)

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
    
    def recreate_mesh(self, n_el=8, fd=shape.circle, h0=0.1, method='greit', parser_meas="rotate_meas", lamb=0.01, p=0.5):
        self.method = method
        self.n_el = n_el
        self.fd = fd
        self.h0 = h0        
        self.mesh_obj = mesh.create(n_el, h0=h0, fd=fd)
        
        self.protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas)
        self.__create_vec_se_to_diff__()

        if method == "bp":
            self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
            self.eit.setup(weight="none")

        elif method == "greit":
            self.eit = greit.GREIT(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=p, lamb=lamb, perm=1, jac_normalized=True)

        elif method == "jac":
            self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
            self.eit.setup(p=p, lamb=lamb, method="kotre", perm=1, jac_normalized=True)
        else:
            raise Exception(f'Method {method} unknown.')

    def setframes(self, Vse, method):
        self.Vse = Vse
        self.Vmeas = self.se_to_diff(Vse)
        self.ds_med_frame = self.eit.solve(self.Vmeas, self.Vref, normalize=True)

        # extract node, element, alpha
        pts = self.mesh_obj.node
        tri = self.mesh_obj.element
        if(method == "jac"):
            self.ds_n = sim2pts(pts, tri, np.real(self.ds_med_frame))

    def updateImage(self, Vse, method,plot_ref=None):
        self.setframes(Vse, method)

        if self.method=='greit':
            x, y ,ds_med_frame = self.eit.mask_value(self.ds_med_frame, mask_value=np.nan) #para 'greit'
            self.image = np.real(ds_med_frame)
            
            if plot_ref!=None:
                plot_ref.set_data(self.image)
        
        elif self.method=='bp':
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element

            self.image = np.real(self.ds_med_frame)

            if plot_ref!=None:
                plot_ref.set_array(self.image)

        elif self.method =='jac':

            # extract node, element, alpha
            pts = self.mesh_obj.node
            tri = self.mesh_obj.element

            self.image = sim2pts(pts, tri, self.ds_med_frame)

            if plot_ref!=None:
                plot_ref.set_array(self.image)

                
        return self.image