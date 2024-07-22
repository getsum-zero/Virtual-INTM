import jax.numpy as jnp
import os
import numpy as np

import ttkbootstrap as ttk
from utils.synapses import synapses
from utils.plane import plane, WeightInit, fromRealdata
# from utils.encoding import On_Off_encoding

import brainpy as bp
import brainpy.math as bm
import tkinter as tk
import pickle
import seaborn as sb

from utils.fig import sigle_save, draw_res
# import gif


class SNN(bp.DynamicalSystem):
    def __init__(self, N, topology, args, neuron_args):
        super().__init__()

        neuron_type = {
            "Izhikevich": bp.neurons.Izhikevich,
            "ExpIF": bp.neurons.ExpIF,
            "GIF": bp.neurons.GIF,
            "LIF": bp.neurons.LIF,
        }
        neuron_imp = None
        try:
            neuron_imp = neuron_type[args["neuron_type"]]
        except Exception as e:
            raise("Neuron model error!")
        
        try:
            self.n = neuron_imp(N, **neuron_args[args["neuron_type"]])
        except:
            self.n = neuron_imp(N)


        
        self.s = synapses(self.n, self.n,
                          conn=topology.cij,
                          synapses_type=args["synapses_type"],

                          plasticity_type = args["plasticity_type"],
                          multiple= args["multiple"],
                          w_bound = args["w_bound"],
                          g_max=WeightInit(topology, args["w_bound"]),
                          homeo_type = args["homeo_type"],
                          dy_delete = args["dy_delete"],

                          stp_dict = args["stp"], 
                          stdp_dict = args["stdp"], 

                )
    def update(self, tdi, x):
        self.s.update(tdi)
        x  >> self.n
        self.s.weight_update(tdi)

    def set_para(self, mod):
        self.s.set_para(mod = mod)
    
    def clear_input(self):
        self.n.clear_input()


# ================= Environment settings ========================
def defult_setting(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args["cuda_devices"])
    
    bm.set_dt(args["dt"])
    bm.set_platform(args["platform"])
    bm.set_environment(bm.training_mode)
# ===============================================================


def getModel(args):
    real_data_args = args["real_world_data"]
    topology_args = args["planar_topology"]
    synapses_args = args["synapses"]
    neuron_args = args["Neuron"]

    np.random.seed(int(args["Running"]["seed"]))

    if real_data_args["mode"] == "fitting":
        real_data = fromRealdata(real_data_args["response"], 
                                shape = (args["shape"]["row"],args["shape"]["col"]), 
                                draw_ori = real_data_args["draw_ori"], 
                                draw_p = real_data_args["draw_p"],
                                stTime = real_data_args["stTime"],
                                cutTime = real_data_args["cutTime"],
                                pseudo_trace = real_data_args["pseudo_trace"],
                                save_path = real_data_args["savepath"]
        )
        real_data_sp = np.sum(real_data, axis=0)
        real_data_sp = real_data_sp / np.sum(real_data_sp)
        real_data_sp = real_data_sp.reshape((args["shape"]["row"],args["shape"]["col"]))
        topology = plane(
                    Ndata = real_data_sp, 
                    num = topology_args["N"], 
                    unit = topology_args["unit"], 
                    ele_scale = topology_args["ele_scale"], 
                    cell_unit = topology_args["cell_unit"],
                    cell_scale = topology_args["cell_scale"],
                    cell_prob = topology_args["cell_prob"],
                    lateral_inh = topology_args["lateral_inh"],

                    connect_len = topology_args["connect_len"], 
                    near_p = topology_args["near_p"],
                    far_n = topology_args["far_n"],
                    fire_n = topology_args["fire_n"],

                    draw_point = topology_args["draw_point"],
                    draw_connet = topology_args["draw_connet"],
                    draw_3D_face = topology_args["draw_3D_face"],
                    savepath = topology_args["savepath"],
                    
                    messager=np.where(np.array(real_data_args["stimulus"])>0)[0]
        )
        
        net = SNN(N = topology_args["N"], topology = topology, args = synapses_args, neuron_args = neuron_args)
        return real_data, topology, net

    elif real_data_args["mode"] == "simulation":
        loadpath = real_data_args["loadpath"]
        topology = plane()
        with open(os.path.join(loadpath, "topology.pkl"), 'rb') as file:
            topology = pickle.load(file)
        print("Loading topology from " + os.path.join(loadpath, "topology.pkl"))
        real_data = None
        if real_data_args["backward"] is not None:
            real_data = fromRealdata(real_data_args["backward"], 
                                shape = (args["shape"]["row"],args["shape"]["col"]), 
                                draw_ori = real_data_args["draw_ori"], 
                                draw_p = real_data_args["draw_p"],
                                stTime = real_data_args["stTime"],
                                cutTime = real_data_args["cutTime"],
                                pseudo_trace = real_data_args["pseudo_trace"],
                                save_path = real_data_args["savepath"]
                            )
        
        net = SNN(N = topology_args["N"], topology = topology, args = synapses_args, neuron_args = neuron_args)
        # states = bp.checkpoints.load_pytree(os.path.join(loadpath, "model.bp"))
        # net.load_state_dict(states["net"])
        return real_data, topology, net
    else:
        raise("Mode error!")


def balance_fire_rate(spikes, real_data, timescale = 1.0):
    tot_fire = np.sum(real_data)
    res = np.zeros_like(spikes)
    for i in range(spikes.shape[0]):
        if np.sum(spikes[:i+1]) >= tot_fire * timescale:
            tot_l = int(np.floor(res.shape[0] / (i+1)))
            more = spikes.shape[0] - tot_l * (i+1)
            xtrick = np.ones(i+1) * tot_l
            xtrick[:more] = xtrick[:more] + 1
            np.random.shuffle(xtrick)
            xtrick = np.cumsum(xtrick) - 1

            res[xtrick.astype(np.int32)] = spikes[:i+1]
            return False, res
    return True, None


def running(args):

    # initial
    N = args["planar_topology"]["N"]
    def getdata():
        return np.array(args["real_world_data"]["stimulus"])
    savepath = args["Running"]["savepath"]
    cons = args["Running"]["cons"]
    warmup = args["Running"]["warmup"]
    during = args["Running"]["during"] + warmup
    epoch = args["Running"]["epoch"]
    interval = args["Running"]["interval"]

    epoch_spike = []

    # preprocess
    for i in range(epoch):
        real_data, topology, net = getModel(args)

        trainer = bp.DSRunner(net, 
                            monitors=['n.spike','n.V'], 
                            data_first_axis = 'T',  
                            progress_bar=True, 
                            # jit=False
        )
        input_MEA = getdata()
        steps = int(1 / bm.get_dt())
        inputdata = bm.zeros(N)
        inputdata[topology.input[1]] = input_MEA[topology.input[0]] * cons
        input_M = bm.zeros(((steps * during), 1, N))
        idx = bm.arange(during) * steps
        input_M[idx,0,:] = inputdata

        net.set_para(mod = True)
        trainer.run(inputs = input_M, reset_state=True)
        
        spikes = np.zeros((during * steps, 64))
        for i in range(topology.input.shape[1]):
            x = topology.input[0,i]
            y = topology.input[1,i]
            spikes[:,x] += trainer.mon["n.spike"][:,0,y]
        
        spikes = spikes[warmup * steps:,:]
        for i in range(0, spikes.shape[0], interval):
            idex = np.sum(spikes[i:i+interval], axis=0) > 0
            if interval > 1:
                spikes[i+1:i+interval] = 0
            spikes[i] = idex
        
        if real_data is not None:
            flag, spikes = balance_fire_rate(spikes, real_data, args["Setting"]["timescale"])
            if flag:
                args["Running"]["during"] = args["Running"]["during"] + 10
                return running(args)
        
        # spikes = np.sum(spikes, axis=0)
        # spikes = spikes / (np.sum(spikes) + 1e-8)
        epoch_spike.append(spikes)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'topology.pkl'), 'wb') as file:
        pickle.dump(topology, file)
    print("Saving Topology into!" + os.path.join(savepath, "topology.yaml")) 

    if args["real_world_data"]["mode"] == "fitting":
        name = "Digital Twin Model"
    else:
        name = "Virtual Experiment"

    sigle_save(real_data, np.array(epoch_spike), savepath=savepath, vis_args=args["Visual"], name = name)
    epoch_spike = np.array(epoch_spike)
    epoch_spike = np.sum(epoch_spike, axis=0) / epoch
    
    return epoch_spike, real_data, trainer.mon["n.V"][:, 0, :]
    # warmup * steps
    
    # bp.checkpoints.save_pytree(os.path.join(savepath, "model.bp"), { 'net': net.state_dict() })

    # plt.pcolormesh(trainer.mon["n.input"][:, 0, :])
    # plt.colorbar()
    # plt.savefig(savepath + "input.png")
    # plt.cla()
    # p = spikes.copy()
    # po = getdata()
    # for i in range(64):
    #     if po[i]: continue
    #     p[:,i] = 0
    #bp.visualize.raster_plot(bm.arange(steps * (during)), p, show=True, color="r")

    # plt.pcolormesh(trainer.mon["n.V"][:, 0, :])
    # plt.colorbar()
    # plt.show()

    

import threading
import time


class TaskWithProgressBar:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.elapsed_time = 0

        label = tk.Label(root, text="Simulation in progress...")
        label.pack(pady=5)
        self.time_label = tk.Label(root, text="Elapsed Time: 0.0s")
        self.time_label.pack(pady=5)
        self.progressbar = ttk.Progressbar(root, mode="indeterminate", length=250, bootstyle="striped", orient="horizontal")
        self.progressbar.pack(pady=20)

        self.start_simulation()
        root.protocol('WM_DELETE_WINDOW', self.no_closing)
    
    def no_closing(self):
        pass


    def start_simulation(self):
        self.display = 1
        self.progressbar.start()
        self.update_time_label()
        self.task_thread = threading.Thread(target=self.run_simu, args=(self.args,))
        self.task_thread.start()
    
    def update_time_label(self):
        self.elapsed_time += 0.1
        if self.display:
            self.time_label.config(text="Elapsed Time: %.2fs" % self.elapsed_time)
            self.root.after(100, self.update_time_label)
    
    def run_simu(self, args):
        try:
            defult_setting(args["Setting"])
            sim, real, v = running(args)
            self.display = 0
            self.progressbar.stop()
            self.root.destroy()
            draw_res(sim, args["Running"]["savepath"], real, args["Running"]["interval"], v, args["Visual"])
            
            # self.time_label.config(text="Simulation Completed in %.2fs" % self.elapsed_time)
            tk.messagebox.showinfo("Message", "Simulation Done!")
        except Exception as e:
            self.progressbar.stop()
            self.root.destroy()
            tk.messagebox.showerror("Error", str(e))
        

 


    
    
