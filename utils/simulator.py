import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import numpy as np

from utils.synapses import synapses
from utils.plane import plane, WeightInit, fromRealdata
# from utils.encoding import On_Off_encoding

import brainpy as bp
import brainpy.math as bm
import tkinter as tk
import pickle
# import gif


class SNN(bp.DynamicalSystem):
    def __init__(self, N, topology, args):
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

    if real_data_args["mode"] == "fitting":
        real_data = fromRealdata(real_data_args["response"], 
                                shape = (8,8), 
                                draw_ori = real_data_args["draw_ori"], 
                                draw_p = real_data_args["draw_p"],
                                save_path = real_data_args["savepath"]
        )
        topology = plane(
                    Ndata = real_data, 
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
        
        net = SNN(N = topology_args["N"], topology = topology, args = synapses_args)
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
            idex = np.sum(spikes[i:i+interval], axis=0) > 0
            if interval > 1:
                spikes[i+1:i+interval] = 0
            spikes[i] = idex
        plt.figure(figsize=(10, 6))
        plt.ion()
        ts = np.arange(spikes.shape[0]) * bm.get_dt()
        i = 0
        # frames = []
        while i < spikes.shape[0]:
            plt.clf()
            plt.subplots_adjust(wspace =0.5, hspace = 0.3)
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
        # gif.save(frames, os.path.join(savepath, "res.gif"), duration=1000)
        # plt.savefig(os.path.join(savepath, "outcomes_sim.png"), dpi = 300)
    
    else:

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(wspace =0.5, hspace = 0.3)
        plt.subplot(2,3,1)
        plt.pcolormesh(real_data.reshape((8,8)).T, cmap="Blues")
        plt.colorbar()

        plt.subplot(2,3,2)
        plt.pcolormesh(spikes.reshape((8,8)).T, cmap="Blues")
        plt.colorbar()

        plt.subplot(2,1,2)

        real_data = real_data.reshape(-1)
        spikes = spikes.reshape(-1)
        x_ticks = np.arange(0, spikes.shape[0])
        width = 0.4
        plt.bar(x_ticks - width/2, real_data, width = width, label='Real-world')
        plt.bar(x_ticks + width/2, spikes, width = width, label='Framework')
        plt.ylabel("Firing rate")
        plt.xlabel("MEA index")
        plt.legend()

        plt.subplot(2,3,3)
        cum_real_data = np.cumsum(real_data)
        cum_spikes = np.cumsum(spikes)
        l = np.arange(cum_real_data.shape[0])
        plt.plot(l, cum_real_data, label='Real-world')
        plt.plot(l, cum_spikes, label='Framework')
        plt.legend()

        # plt.subplot(1,4,4)
        # plt.plot(np.arange(64), real_data.reshape(-1), label = "Real-world")
        # plt.plot(np.arange(64), spikes.reshape(-1), label = "Framework")
        
        plt.savefig(os.path.join(savepath, "outcomes.png"), dpi = 300)
        plt.show()
        plt.close()




def running(real_data, topology, net, args):

    trainer = bp.DSRunner(net, 
                          monitors=['n.spike','n.V'], 
                          data_first_axis = 'T',  
                          progress_bar=True, 
                          # jit=False
    )
    # initial
    N = args["planar_topology"]["N"]
    def getdata():
        return np.array(args["real_world_data"]["stimulus"])
    savepath = args["Running"]["savepath"]
    cons = args["Running"]["cons"]
    during = args["Running"]["during"]  
    epoch = args["Running"]["epoch"]
    interval = args["Running"]["interval"]

    # preprocess
    input_MEA = getdata()
    steps = int(1 / bm.get_dt())
    inputdata = bm.zeros(N)
    inputdata[topology.input[1]] = input_MEA[topology.input[0]] * cons
    input_M = bm.zeros(((steps * during), 1, N))
    idx = bm.arange(during) * steps
    input_M[idx,0,:] = inputdata

    epoch_spike = []

    for i in range(epoch):
        net.set_para(mod = True)
        trainer.run(inputs = input_M, reset_state=True)
        
        spikes = np.zeros((during * steps, 64))
        for i in range(topology.input.shape[1]):
            x = topology.input[0,i]
            y = topology.input[1,i]
            spikes[:,x] += trainer.mon["n.spike"][:,0,y]
        
        for i in range(0, spikes.shape[0], interval):
            idex = np.sum(spikes[i:i+interval], axis=0) > 0
            if interval > 1:
                spikes[i+1:i+interval] = 0
            spikes[i] = idex

        spikes = np.sum(spikes, axis=0)
        spikes = spikes / (np.sum(spikes) + 1e-8)
        epoch_spike.append(spikes)
        
    epoch_spike = np.array(epoch_spike)
    epoch_spike = np.sum(epoch_spike, axis=0) / epoch
    
    
    with open(os.path.join(savepath, 'topology.pkl'), 'wb') as file:
        pickle.dump(topology, file)
    print("Saving Topology into!" + os.path.join(savepath, "topology.yaml"))
    bp.checkpoints.save_pytree(os.path.join(savepath, "model.bp"), { 'net': net.state_dict() })

    draw_res(epoch_spike, savepath, real_data, interval, trainer.mon["n.V"][:, 0, :])
   

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

    



def simulate(args, root):
    
    print(args)
    # rf = open(file="./config/args.yaml", mode='r')
    # crf = rf.read()
    # rf.close()
    # args = yaml.load(stream=crf, Loader=yaml.FullLoader)
    try:
        def on_close():
            popup.destroy()

        popup = tk.Toplevel(root)
        popup.title("Simulation Progress")
        popup.geometry("300x100")
        popup.protocol("WM_DELETE_WINDOW", on_close)

        label = tk.Label(popup, text="Simulation in progress...")
        label.pack(pady=20)
        root.update()

        defult_setting(args["Setting"])
        real_data, topology, net = getModel(args)
        
        running(real_data, topology, net, args)
        popup.destroy()
        tk.messagebox.showinfo("Message", "Simulation Done!")
    except Exception as e:
        popup.destroy()
        tk.messagebox.showerror(title='Error', message=str(e))
    
