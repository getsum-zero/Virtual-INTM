import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
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

from utils.fig import sigle_save
# import gif

simu_color = "#608BDF"
real_color = "#D8D8D8"


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

    if real_data_args["mode"] == "fitting":
        real_data = fromRealdata(real_data_args["response"], 
                                shape = (args["shape"]["row"],args["shape"]["col"]), 
                                draw_ori = real_data_args["draw_ori"], 
                                draw_p = real_data_args["draw_p"],
                                stTime = real_data_args["stTime"],
                                cutTime = real_data_args["cutTime"],
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
        
        net = SNN(N = topology_args["N"], topology = topology, args = synapses_args)
        # states = bp.checkpoints.load_pytree(os.path.join(loadpath, "model.bp"))
        # net.load_state_dict(states["net"])
        return None, topology, net
    else:
        raise("Mode error!")



def dyn_draw(i, spikes, time, index, x_ticks, prV, r, spikes_, width):
    plt.subplot(2,3,1)
    plt.xlim(0, bm.get_dt() * (i+1))
    plt.ylim(0, spikes.shape[1])
    plt.scatter(time, index, marker=".", s=8, color =(75/255, 101/255, 175/255))
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")

    plt.subplot(2,3,2)
    
    plt.pcolormesh(prV.T)
    plt.xlabel("Time ({:.2f}s)".format(bm.get_dt()))
    plt.ylabel("Neuron index")
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.pcolormesh(np.sum(spikes[i:r], axis=0).reshape((8,8)).T, cmap="Blues")

    

    plt.subplot(2,1,2)
    plt.bar(x_ticks + width/2, spikes_, width = width, label='Framework', color = (75/255, 101/255, 175/255))
    plt.xlim(0, spikes.shape[1])
    plt.ylim(0, np.max(spikes_)+1)
    plt.ylabel("Spike count")
    plt.xlabel("MEA index")
    plt.suptitle('Time: {:.2f}s'.format(r * bm.get_dt()), fontsize=15)

    plt.pause(0.0001)


# @gif.frame
# def dyn_draw_gif(i, spikes, time, index, x_ticks, prV, r, spikes_, width):
#     dyn_draw(i, spikes, time, index, x_ticks, prV, r, spikes_, width)


def draw_res(spikes, savepath, real_data, interval, Vmat):

    # bp.visualize.raster_plot(bm.arange(spikes.shape[0]) , spikes, show=False)
    # plt.savefig(savepath + "spikes.png", dpi = 300)
    # plt.close()

    if real_data is None:
        for i in range(0, spikes.shape[0], interval):
            idex = np.sum(spikes[i:i+interval], axis=0) >= 0.5
            if interval > 1:
                spikes[i+1:i+interval] = 0
            spikes[i] = idex
        
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(wspace =0.5, hspace = 0.3)
        plt.ion() 
        ts = np.arange(spikes.shape[0]) * bm.get_dt()
        i = 0
        # frames = []
        while i < spikes.shape[0]:
            plt.clf()
            r = min(int(i + 1 / bm.get_dt() / 4), spikes.shape[0])
            if np.sum(spikes[i:r])  < 15:
                i = r
            elements = np.where(spikes[:i+1] > 0.)
            index = elements[1]
            time = ts[elements[0]]
            prV = Vmat[:i]

            x_ticks = np.arange(0, spikes.shape[1])
            spikes_ = np.sum(spikes[:r], axis=0).reshape(-1)
            width = 0.4

            # frames.append(dyn_draw_gif(i, spikes, time, index, x_ticks, prV, r, spikes_, width))
            dyn_draw(i, spikes, time, index, x_ticks, prV, r, spikes_, width)
            i = min(i + 10, spikes.shape[0])

        plt.ioff()
        plt.show()
        plt.close()

        # gif.save(frames, os.path.join(savepath, "res.gif"), duration=1000)
        # plt.savefig(os.path.join(savepath, "outcomes_sim.png"), dpi = 300)
    
    else:

        V = Vmat
        # V = trainer.mon["n.V"][:, 0, :]
        fig, ax = plt.subplots()
        colp = ax.pcolormesh(V, )#cmap = "RdBu_r")
        plt.colorbar(colp)
        ax.set_title("Membrane potential")
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Time (s)")
        ylabels = ax.get_yticks().tolist()
        ylabels = (np.array(ylabels) * bm.get_dt()).tolist()
        ax.set_yticklabels(ylabels)
        plt.savefig(os.path.join(savepath, "V.png"), dpi = 300)
        plt.close()
        # fig, ax = plt.subplots()
        # colp = ax.pcolormesh(V)
        # plt.colorbar(colp)
        # ax.set_title("Membrane potential")
        # ax.set_xlabel("Neuron index")
        # ax.set_ylabel("Time (s)")
        # ylabels = ax.get_yticks().tolist()
        # ylabels = (np.array(ylabels) * bm.get_dt()).tolist()
        # ax.set_yticklabels(ylabels)
        # plt.savefig(os.path.join(savepath, "V.png"), dpi = 300)
        # plt.close()


        real_data = np.sum(real_data, axis=0) / np.sum(real_data)
        spikes = np.sum(spikes, axis=0) / np.sum(spikes)


        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(wspace =0.5, hspace = 0.3)
        plt.subplot(2,3,1)
        sb.heatmap(real_data.reshape((8,8)),  cmap="Blues", cbar_kws={'label': 'Normalized fire rate',})
        plt.title("Real-world")
        plt.xlabel("Electrodes Col")
        plt.ylabel("Electrodes Row")
        plt.xticks([])
        plt.yticks([])

        # loc = [0, times / 2, times]
        # lab = [0, spikes.shape[0] * bm.get_dt() / 2, spikes.shape[0] * bm.get_dt()]
        # locy = np.arange(spikes.shape[1] // 8) * 8
        # plt.xticks(loc, lab)
        # plt.yticks(locy, locy)



        plt.subplot(2,3,2)
        sb.heatmap(spikes.reshape((8,8)), cmap="Blues", cbar_kws={'label': 'Normalized fire rate', })
        plt.title("Framework")
        plt.xlabel("Electrodes Col")
        plt.ylabel("Electrodes Row")
        plt.xticks([])
        plt.yticks([])


        plt.subplot(2,3,3)
        real_data = real_data / np.sum(real_data)
        spikes = spikes / np.sum(spikes)
        cum_real_data = np.cumsum(real_data)
        cum_spikes = np.cumsum(spikes)

        l = np.arange(cum_real_data.shape[0])
        plt.plot(l, cum_real_data, label='Real-world')
        plt.plot(l, cum_spikes, label='Framework')
        plt.legend()


        plt.subplot(2,1,2)
        real_data = real_data.reshape(-1)
        spikes = spikes.reshape(-1)
        x_ticks = np.arange(0, spikes.shape[0])
        width = 0.6
        plt.bar(x_ticks, spikes, width = width, label='Framework',  edgecolor='white', color = simu_color)
        plt.bar(x_ticks, -real_data, width = width, label='Real-world', edgecolor='white', color = real_color)
        # for x, y in zip(x_ticks, real_data):
        #     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
        plt.ylabel("Normalized fire rate")
        plt.xlabel("MEA index")
        plt.legend()
        maxx = max(np.max(real_data), np.max(spikes)) / 4
        idx = (np.arange(9) - 4) * maxx
        ytext = ['%.2f' % x for x in np.abs(idx)]
        plt.yticks(idx, ytext)

        # plt.subplot(1,4,4)
        # plt.plot(np.arange(64), real_data.reshape(-1), label = "Real-world")
        # plt.plot(np.arange(64), spikes.reshape(-1), label = "Framework")
        
        plt.savefig(os.path.join(savepath, "outcomes.png"), dpi = 300)
        plt.show()
        plt.close()




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
        # spikes = np.sum(spikes, axis=0)
        # spikes = spikes / (np.sum(spikes) + 1e-8)
        epoch_spike.append(spikes)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'topology.pkl'), 'wb') as file:
        pickle.dump(topology, file)
    print("Saving Topology into!" + os.path.join(savepath, "topology.yaml"))  

    sigle_save(real_data, np.array(epoch_spike), savepath=savepath, color = {"bar": (real_color, simu_color)})
    epoch_spike = np.array(epoch_spike)
    epoch_spike = np.sum(epoch_spike, axis=0) / epoch
    
    return epoch_spike, real_data, trainer.mon["n.V"][warmup * steps:, 0, :]
    
    
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
        # try:
        defult_setting(args["Setting"])
        sim, real, v = running(args)
        self.display = 0
        self.progressbar.stop()
        self.root.destroy()
        draw_res(sim, args["Running"]["savepath"], real, args["Running"]["interval"], v)
        
        # self.time_label.config(text="Simulation Completed in %.2fs" % self.elapsed_time)
        tk.messagebox.showinfo("Message", "Simulation Done!")
        # except Exception as e:
        #     self.progressbar.stop()
        #     self.root.destroy()
        #     tk.messagebox.showerror("Error", "Runtime error, please select correct parameters or check the running environment")
        

 


    
    
