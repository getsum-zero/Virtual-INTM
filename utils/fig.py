
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import brainpy as bp
import brainpy.math as bm


simu_color = "#608BDF"
real_color = "#D8D8D8"

def heatmap_with_time(savepath, real_data, spikes, times):
    print(spikes.shape)
    print(real_data.shape)
    assert real_data.shape[0] >= spikes.shape[0]

    real_data_time = np.zeros((times, spikes.shape[1]))
    sim_data_time = np.zeros((times, spikes.shape[1]))
    step_real = int(real_data.shape[0] / times)
    step_sim = int(spikes.shape[0] / times)

    real_now = np.zeros(spikes.shape[1])
    sim_now = np.zeros(spikes.shape[1])
    for i in range(times):
        real_now += np.sum(real_data[i*step_real:(i+1)*step_real, :], axis=0)
        sim_now += np.sum(spikes[i*step_sim:(i+1)*step_sim, :], axis=0)
        real_data_time[i] = real_now / step_real
        sim_data_time[i] = sim_now / step_sim

    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(wspace =0.5, hspace = 0.3)
    plt.subplot(2,3,1)
    sns.heatmap(real_data_time.T,  cmap="RdBu_r")
    plt.title("Real-world")
    plt.xlabel("Time (s)")
    plt.ylabel("Electrodes")
    loc = [0, times / 2, times]
    lab = [0, spikes.shape[0] * bm.get_dt() / 2, spikes.shape[0] * bm.get_dt()]
    locy = np.arange(spikes.shape[1] // 8) * 8
    plt.xticks(loc, lab)
    plt.yticks(locy, locy)



    plt.subplot(2,3,2)
    sns.heatmap(sim_data_time.T, cmap="RdBu_r")
    plt.title("Framework")
    plt.xlabel("Time (s)")
    plt.ylabel("Electrodes")
    plt.xticks(loc, lab)
    plt.yticks(locy, locy)

    
    real_data = np.sum(real_data, axis=0) / np.sum(real_data)
    spikes = np.sum(spikes, axis=0) / np.sum(spikes)

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
    width = 0.4
    plt.bar(x_ticks - width/2, real_data, width = width, label='Real-world')
    plt.bar(x_ticks + width/2, spikes, width = width, label='Framework')
    plt.ylabel("Firing rate")
    plt.xlabel("MEA index")
    plt.legend()

    # plt.subplot(1,4,4)
    # plt.plot(np.arange(64), real_data.reshape(-1), label = "Real-world")
    # plt.plot(np.arange(64), spikes.reshape(-1), label = "Framework")
    
    plt.savefig(os.path.join(savepath, "outcomes.png"), dpi = 300)
    plt.show()
    plt.close()




def spikes_time(spikes, name, savepath):
    steps = int(1/bm.get_dt())
    bp.visualize.raster_plot(bm.arange(spikes.shape[0] * steps) / steps, spikes, show=False, xlabel='Time (s)', ylabel="MEA index", title='Framework')
    plt.savefig(os.path.join(savepath, name + "_spikes.png"), dpi = 300)
    plt.close()

    Marginal_Histogram(spikes, bm.get_dt(), name, 
                       os.path.join(savepath, name + "spikes_hist.png"))


def Marginal_Histogram(data, dt, name, path):  # 0, 1  (Time, MEA_index)

    fig = plt.figure(figsize=(16, 10), dpi = 300)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    x, y = np.where(data > 0)

    # Scatterplot on main ax
    ax_main.scatter(x, y, s = 5, c = "black")

    # histogram on the right
    ax_bottom.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
    ax_bottom.invert_yaxis()

    # histogram in the bottom
    ax_right.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')

    # Decorations
    ax_main.set(title=name, xlabel='Time(s)', ylabel='MEA index')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)

    xlabels = ax_main.get_xticks().tolist()
    xlabels = (np.array(xlabels) * dt).tolist()
    ax_main.set_xticklabels(xlabels)
    plt.savefig(path, dpi = 300)
    plt.close()



def sigle_save(real_data_o, spikes_o, savepath, vis_args):

    font = vis_args["font"]
    color_bar = vis_args["bar"]

    plt.rcParams['font.sans-serif'] = font


    if real_data_o is None:
        return 
    
    real_spikes = real_data_o.copy()
    spikes_trains = spikes_o.copy()

    minlen = min(real_data_o.shape[0], spikes_o.shape[1])

    # spike
    steps = int(1/bm.get_dt())
    bp.visualize.raster_plot(bm.arange(minlen * steps) / steps, spikes_o[0, :minlen], show=False, xlabel='Time (s)', ylabel="Electrodes", title='Framework')
    plt.savefig(os.path.join(savepath, "sim_spikes.png"), dpi = 300)
    plt.close()

    steps = int(1/bm.get_dt())
    bp.visualize.raster_plot(bm.arange(minlen * steps) / steps, real_data_o[:minlen], show=False, xlabel='Time (s)', ylabel="Electrodes", title='Real-world')
    plt.savefig(os.path.join(savepath, "real_spikes.png"), dpi = 300)
    plt.close()

    # cumsum line
    xtrick = np.arange(real_spikes.shape[1])
    fig, ax = plt.subplots()
    real_data = np.mean(real_spikes, axis=0)
    # ax.grid(True, alpha=0.3)
    if np.ndim(spikes_trains) > 2:
        spikes =  np.mean(np.mean(spikes_trains, axis=0), axis=0)
        spikes = spikes / np.sum(spikes)
        l, = ax.plot(xtrick, np.cumsum(spikes), label='Framework')
        cum_real_temp = np.cumsum(np.sum(spikes_trains, axis = 1), axis=1)
        cum_real = np.array([x/np.max(x) for x in cum_real_temp])
        upbound = np.max(cum_real, axis=0)
        lowbound = np.min(cum_real, axis=0)
        print(upbound, lowbound)
        ax.fill_between(xtrick,
                    lowbound, upbound,
                    color=l.get_color(), alpha=.3)
    else:
        spikes = np.mean(spikes_trains, axis=0)
        spikes = spikes / np.sum(spikes)
        ax.plot(xtrick, np.cumsum(spikes), label='Framework')

    real_data = real_data / np.sum(real_data)
    ax.plot(xtrick, np.cumsum(real_data), label='Real-world')
    ax.set_xlabel("Index")
    ax.set_ylabel("Normalized fire rate")
    plt.legend()
    plt.savefig(os.path.join(savepath, "line.png"), dpi = 300)
    plt.close()

   

    # =================================  heatmap ==============================
    maxx = max(np.max(real_data), np.max(spikes))
    # heatmap of real data
    sns.heatmap(real_data.reshape((8,8)).T, square=True, annot=True, cmap='Blues', annot_kws={'size': 5}, 
                vmin = 0, vmax=maxx, cbar_kws={'label': 'Normalized fire rate', "shrink": 0.8})
    plt.title('The proportion of spikes with real-world')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(savepath, "real_heatmap.png"), dpi = 300)
    plt.close()
    # heatmap of framework
    sns.heatmap(spikes.reshape((8,8)).T, square=True, annot=True, cmap='Blues', annot_kws={'size': 5},
                vmin = 0, vmax=maxx, cbar_kws={'label': 'Normalized fire rate', "shrink": 0.8})
    plt.title('The proportion of spikes with framework')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(savepath, "frame_heatmap.png"), dpi = 300)
    plt.close()



    
    # fire rate across MEA
    plt.figure(figsize=(10, 4))
    real_data = real_data.reshape(-1)
    spikes = spikes.reshape(-1)
    real_data = real_data / np.sum(real_data)
    spikes = spikes / np.sum(spikes)
    x_ticks = np.arange(0, spikes.shape[0])
    width = 0.4

    plt.bar(x_ticks - width/2, spikes, width = width, label='Framework',  edgecolor='white', color = color_bar["simu_color"])
    plt.bar(x_ticks + width/2, real_data, width = width, label='Real-world', edgecolor='white', color = color_bar["real_color"])
    # for x, y in zip(x_ticks, real_data):
    #     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylabel("Normalized fire rate")
    plt.xlabel("MEA index")
    plt.legend()
    # maxx = max(np.max(real_data), np.max(spikes)) / 4
    # idx = (np.arange(9) - 4) * maxx
    # ytext = ['%.2f' % x for x in np.abs(idx)]
    # plt.yticks(idx, ytext)
    plt.savefig(os.path.join(savepath, "hist_across_mea.png"), dpi = 300)
    plt.close()


    # fire rate across time
    real_mea = np.sum(real_spikes, axis=1)
    spikes_mea = np.mean(np.sum(spikes_trains, axis=2), axis=0)
    interval_sim = int(0.2 / bm.get_dt())
    num = int(spikes_mea.shape[0] / interval_sim)
    interval_real = int(real_mea.shape[0] / num)

    real_mea_i = np.zeros(num)
    spikes_mea_i = np.zeros(num)

    for i in range(num):
        real_mea_i[i] = np.sum(real_mea[i*interval_real:(i+1)*interval_real])
        spikes_mea_i[i] = np.sum(spikes_mea[i*interval_sim:(i+1)*interval_sim])


    real_mea_i = real_mea_i / np.sum(real_mea_i)
    spikes_mea_i = spikes_mea_i / np.sum(spikes_mea_i)
    plt.figure(figsize=(10, 4))
    x_ticks = np.arange(0, num) * 0.1
    width = 0.04
    
    plt.bar(x_ticks + width/2, spikes_mea_i, width = width, label='Framework',  edgecolor='white', color = color_bar["simu_color"])
    plt.bar(x_ticks - width/2, real_mea_i, width = width, label='Real-world', edgecolor='white', color = color_bar["real_color"])
    # plt.figure(figsize=(10, 4))
    # real_mea_x, _ = np.where(real_spikes)
    # spikes_mea_x, _ = np.where(spikes_trains)
    # interval = int(0.1 / bm.get_dt())
    # real_mea_x = np.round(real_mea_x / interval) * interval
    # spikes_mea_x = np.round(spikes_mea_x / interval) * interval

    # real_mea = pd.DataFrame(real_mea_x.reshape(-1,1), columns=["Real-world"])
    # spikes_mea = pd.DataFrame(spikes_mea_x.reshape(-1,1), columns=["Framework"])

    # fig,axes=plt.subplots() 
    # sns.histplot(pd.DataFrame(real_mea), ax=axes, kde=True)
    # sns.histplot(pd.DataFrame(spikes_mea), ax=axes, kde=True)
    plt.ylabel("Normalized fire rate")
    plt.xlabel("Time (s)")
    plt.legend()
    # maxx = max(np.max(real_mea_i), np.max(spikes_mea_i)) / 4
    # idx = (np.arange(9) - 4) * maxx
    # ytext = ['%.2f' % x for x in np.abs(idx)]
    # plt.yticks(idx, ytext)
    plt.savefig(os.path.join(savepath, "hist_across_time.png"), dpi = 300)
    plt.close()


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


def save_data(savepath, real_data, spikes):
    try:
        save_path = os.path.join(savepath, "spikes/")
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        np.save(os.path.join(save_path, "real_data.npy"), real_data)
        np.save(os.path.join(save_path, "sim_data.npy"), spikes)
        print("Saved !")
    except:
        raise("Fail to save")


def draw_res(spikes, savepath, real_data, interval, Vmat, vis_args):

    font = vis_args["font"]
    color_bar = vis_args["bar"]
    plt.rcParams['font.sans-serif'] = font

    save_data(savepath, real_data, spikes)

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
        
        # draw V
        V = Vmat
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


        # draw heatmap
        real_data = np.sum(real_data, axis=0) / np.sum(real_data)
        spikes = np.sum(spikes, axis=0) / np.sum(spikes)
        maxx = max(np.max(real_data), np.max(spikes))
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.subplot(2,3,1)
        sns.heatmap(real_data.reshape((8,8)), cmap="Blues", vmin = 0, vmax = maxx,
                    cbar_kws={'label': 'Normalized fire rate', "shrink": 0.8})
        plt.title("Real-world")
        plt.xlabel("Electrodes Col")
        plt.ylabel("Electrodes Row")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,3,2)
        sns.heatmap(spikes.reshape((8,8)), cmap="Blues", vmin = 0, vmax = maxx,
                    cbar_kws={'label': 'Normalized fire rate', "shrink": 0.8})
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
        width = 0.4
        plt.bar(x_ticks - width / 2 , real_data, width = width, label='Real-world', edgecolor='white', color = color_bar["real_color"])
        plt.bar(x_ticks + width / 2, spikes, width = width, label='Framework',  edgecolor='white', color = color_bar["simu_color"])
        
        # for x, y in zip(x_ticks, real_data):
        #     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
        plt.ylabel("Normalized fire rate")
        plt.xlabel("MEA index")
        plt.legend()
        # maxx = max(np.max(real_data), np.max(spikes)) / 4
        # idx = (np.arange(9) - 4) * maxx
        # ytext = ['%.2f' % x for x in np.abs(idx)]
        # plt.yticks(idx, ytext)
        # plt.subplot(1,4,4)
        # plt.plot(np.arange(64), real_data.reshape(-1), label = "Real-world")
        # plt.plot(np.arange(64), spikes.reshape(-1), label = "Framework")
        
        plt.savefig(os.path.join(savepath, "outcomes.png"), dpi = 300)
        plt.show()
        plt.close()




