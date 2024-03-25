# -*- coding: utf-8 -*-

'''
For topological connections

'''

from typing import Dict, List, Union, Sequence, Callable, Optional

import brainpy as bp
import brainpy.math as bm
from brainpy.types import ArrayType, Shape

import jax.numpy as jnp

from brainpy._src.dynsys import NeuGroup, NeuGroupNS, TwoEndConn
from brainpy._src.synouts import COBA
from brainpy._src.initialize import variable
from brainpy._src.dyn.base import SynDyn
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint

class DelayNeu(NeuGroupNS):
  def __init__(
      self,
      size,
      delay_len,
      host,

      keep_size = False,
      mode = None,
      name = None
  ):
    super(DelayNeu, self).__init__(name=name,
                                    size=size,
                                    keep_size=keep_size,
                                    mode=mode)
    self.host = host
    self.delay_len = delay_len
    self.delay = {}
    self.spike = variable(bm.zeros, self.mode, self.varshape)
    self.timestamp = 0

  def update(self, x=None):
    self.delay[self.timestamp] = self.host.spike.copy()
    id = self.timestamp - self.delay_len
    self.spike = self.delay.get(id, bm.zeros_like(self.host.spike))
    self.delay.pop(id, 0)
    self.timestamp = self.timestamp + 1
    # print(self.delay)
    return self.spike
    
  def reset_state(self, batch_size=None):
    self.delay.clear()
    self.spike = variable(bm.zeros, batch_size, self.varshape)
    self.timestamp = 0

 
'''
STDP:
    batch (mean): done
'''

class plasticity_stdp(bp.DynamicalSystem):

    def __init__(
        self,
        pre,
        post,
        conn,
        
        stdp_dict: dict = None,
        **kwargs
    ):
        super().__init__()

        self.pre = pre
        self.post = post
        self.conn = conn

        self.tau_s = stdp_dict.get("tau_s", 40)
        self.tau_t = stdp_dict.get("tau_t", 40)
        self.A1 = stdp_dict.get("A1", 1)
        self.A2 = stdp_dict.get("A2", 1)
        self.theta_p = stdp_dict.get("theta_p", 1e-3)
        self.theta_n = stdp_dict.get("theta_n", 1e-3)


        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        self.num = len(self.pre_ids)
        self.Wshape = self.conn.require('conn_mat').shape

        self.trace_pre = variable(bm.zeros, self.mode, self.num)
        self.trace_post = variable(bm.zeros, self.mode, self.num)
        self.integral_stdp = bp.odeint(method=stdp_dict.get("method", 'exp_auto'), f=self.derivative_stdp)

    @property
    def derivative_stdp(self):
        dtrace_pre = lambda trace_pre, t: - trace_pre / self.tau_s
        dtrace_post = lambda trace_post, t: - trace_post / self.tau_t
        return bp.JointEq([dtrace_pre, dtrace_post]) 
    
    def reset_state(self, batch_size=None):
        self.trace_pre = variable(bm.zeros, batch_size, self.num)
        self.trace_post = variable(bm.zeros, batch_size, self.num)
    
    def _trace_update(self, pre_spikes, post_spikes, t, dt):
        self.trace_pre.value, self.trace_post.value = self.integral_stdp(self.trace_pre, self.trace_post, t, dt)
        trace_pre = bm.where(pre_spikes, self.trace_pre + self.A1, self.trace_pre)
        trace_post = bm.where(post_spikes, self.trace_post + self.A2, self.trace_post)

        self.trace_pre.value = trace_pre
        self.trace_post.value = trace_post
    
    def get_weight(self, tdi):
        pre_spikes = self.pre.spike.value[:, self.pre_ids].reshape(-1, self.num)
        post_spikes = self.post.spike.value[:, self.post_ids].reshape(-1, self.num)

        if len(pre_spikes.shape) == 2:
            batch_size = pre_spikes.shape[0]
        else:
            batch_size = 1

        w_pre = jnp.sum(bm.where(pre_spikes, -self.trace_post , 0).value, axis=0)
        w_post = jnp.sum(bm.where(post_spikes, self.trace_pre, 0).value, axis=0)
        res = None
        if isinstance(self.conn, bp.connect.One2One):
            res = w_pre * self.theta_n / batch_size + w_post * self.theta_p / batch_size
        else:
            delta_w_pre = bm.zeros(self.Wshape)
            delta_w_post = bm.zeros(self.Wshape)
            delta_w_pre[self.pre_ids, self.post_ids] = w_pre
            delta_w_post[self.pre_ids, self.post_ids] = w_post
            res = delta_w_pre * self.theta_n / batch_size + delta_w_post * self.theta_p / batch_size
        
        self._trace_update(pre_spikes, post_spikes, tdi.t, tdi.dt)

        return res


'''
multiple synapses (delay): done

dynamic connection:
    delete synapses : w = 0 => deleted

    

note:
    * By default all synaptic weights are positive
'''
    

class synapses(bp.DynamicalSystem):

    def __init__(
            self,
            pre,
            post,
            conn,
            synapses_type,
            multiple: List = [0, ],

            plasticity_type: str = "STDP",
            mod: bool = False,   # weight update flag

            eps: float = 1e-9,
            w_bound: float = 1,
            homeo_type: str = "norm",
            dy_delete: bool = True,

            stp_dict: dict = None,
            stdp_dict: dict = None,
            **kwargs
        ):
        super().__init__()
        self.conn = conn
        self.multiple = multiple
        self.mod = mod

        self.delet_bound = eps
        self.w_bound = w_bound
        self.homeo_type = homeo_type
        self.dy_delete = dy_delete

        self.DelayNeu = []
        self.synapses = []
        self.STP = []
        self.plasticity = []
        
        synapses_imp = None
        synapses_list = {
            "AMPA": bp.synapses.AMPA,  # ok
            "GABAa": bp.synapses.GABAa,
            "BioNMDA": bp.synapses.BioNMDA, 
            "NMDA": bp.synapses.NMDA,
        }
        try:
            synapses_imp = synapses_list[synapses_type]
        except:
            raise("Error synapses type!")

  

        for i, s in enumerate(multiple):
            # delay neurons
            self.DelayNeu.append( DelayNeu(size=pre.varshape, delay_len=s, host=pre) )

            # STP
            self.synapses.append( 
                synapses_imp(self.DelayNeu[i], post, conn, 
                              stp = bp.synapses.STP(U=stp_dict.get("U", 0.2), 
                                                    tau_d=stp_dict.get("tau_d", 2), 
                                                    tau_f=stp_dict.get("tau_f", 2)), 
                              **kwargs) 
            )
            if plasticity_type == "STDP":
                self.plasticity.append( plasticity_stdp(self.DelayNeu[i], post, conn, stdp_dict=stdp_dict) )

            

        print(self.synapses)


    @staticmethod
    def _homeostasis(synapse, bound=1.0, type="norm"):
        if isinstance(synapse.conn, bp.connect.One2One):
            synapse.g_max.value = bm.clip(synapse.g_max, 0, bound)
        else:
            if type == "exp":
                sign = bm.sign(synapse.g_max)
                exp_value = bm.exp(bm.abs(synapse.g_max))
                exp_value = bm.where(synapse.g_max == 0, 0, exp_value)
                sum_value = bm.tile(bm.sum(exp_value, axis = 0), exp_value.shape[0]).reshape(exp_value.shape)
                synapse.g_max.value = exp_value / sum_value * bound * sign
                
            elif type == "norm":
                norm_value = synapse.g_max ** 2
                sum_value = bm.sqrt(bm.sum(norm_value, axis = 0))
                sum_value = bm.tile(sum_value, norm_value.shape[0]).reshape(norm_value.shape)
                synapse.g_max.value = synapse.g_max / sum_value * bound
            elif type == "none": 
                return 
            else:
                raise("Error homeostasis type!")
            

    

    #
    #    external interface
    #


    def set_para(self, mod):
        self.mod = mod
    
    def reset_state(self, batch_size=None):
        for i in range(len(self.multiple)):
            self.DelayNeu[i].reset_state(batch_size)
            self.synapses[i].reset_state(batch_size)
            self.plasticity[i].reset_state(batch_size)
    
    def update(self, tdi, pre_spike=None):
        for i in range(len(self.multiple)):
            self.DelayNeu[i].update(tdi)
            self.synapses[i].update(tdi, pre_spike)

    def weight_update(self, tdi):
        if not self.mod:
            return 
        for i in range(len(self.multiple)):
            # get `dw` by STDP
            wdelta = self.plasticity[i].get_weight(tdi)

            # ================= delete =================
            # For synapses with too small weights, 
            # set dw = 0 and no longer change their weights (delete)
            # to do: 
            #       Actual deletion, freeing up memory
            if self.dy_delete:
                wdelta.value = bm.where(self.synapses[i].g_max > self.delet_bound, wdelta, 0)
            
            # ================= learning =================
            self.synapses[i].g_max.value = self.synapses[i].g_max + wdelta

            # ================= homeostasis =================
            self._homeostasis(self.synapses[i], self.w_bound, self.homeo_type)


            
