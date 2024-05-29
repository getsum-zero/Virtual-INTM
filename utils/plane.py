import numpy as np
import brainpy as bp
import brainpy.math as bm
from scipy import stats
import matplotlib.pyplot as plt
import scipy.io as scio
import os


'''
find method:
    https://cloud.tencent.com/developer/ask/sof/74365
'''


'''
    Ndata:  The probability density is distributed at the grid points
    -------------------------
    |    .      |           |
    |           |           |
    |           |      .    |
    |           |           |
    -------------------------

    step:
        1. the probability density determines which grid the data belongs to
        2. coordinates of random generated points (within the grid)

'''

ele_color = "#D8D8D8"
neur_trans_color = "#608BDF"
neur_color = "#F7E1ED"

def fromRealdata(path, shape, 
                 draw_ori = False, 
                 draw_p = False,
                 save_path = "./log/",
                 stTime = 0,
                 cutTime = 10,
                 ):
    #record = h5py.File(path)
    record = scio.loadmat(path)

    dt = bm.get_dt()
    steps = int(1 / bm.get_dt())
    total = steps * 500
    spikes = np.zeros((total, np.prod(shape)))
    stTime = stTime * steps
    cutTime = cutTime * steps

    isExists=os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path) 

    try:
        for key, val in record.items():
            if key[-2:] != "nr": continue
            if key[11:13].isdigit() == False:  continue 
            data = np.array(val)
            pos = int(key[11:13]) #AnSt_Label_
            pos = (pos // 10 - 1) * 8 + pos % 10 - 1
            data = np.unique(np.round(data / dt)).astype(np.int32)
            spikes[data[data <= total], pos] = 1
        print(np.sum(spikes))
        assert np.sum(spikes) > 0
    except:    
        for key, val in record.items():
            if key[-2:] != "nr": continue
            if key[19:21].isdigit() == False:  continue 
            data = np.array(val)
            pos = int(key[19:21]) #AnSt_Label_E_00159_
            pos = (pos // 10 - 1) * 8 + pos % 10 - 1
            data = np.unique(np.round(data / dt)).astype(np.int32)
            spikes[data[data <= total], pos] = 1
        
    spikes = spikes[stTime:cutTime+stTime]

    p = np.sum(spikes, axis=0)
    p = p / np.sum(p)
    p = p.reshape(shape)
    if draw_ori:
        bp.visualize.raster_plot(bm.arange(cutTime), spikes, show=False, xlabel='Time (s)', ylabel="MEA index", title='Organoid')
        plt.savefig(os.path.join(save_path, "real_data_spikes.png"), dpi = 300)
        plt.cla()
        plt.close()
        #bp.visualize.raster_plot(bm.arange(cutTime), spikes, show=True)
    
    if draw_p:
        plt.pcolormesh(p.reshape(shape).T, cmap="Blues")
        plt.colorbar()
        plt.savefig(os.path.join(save_path , "real_data_firing_rate.png"), dpi = 300)
        plt.cla()
        plt.close()

    return spikes
        


class WeightInit(bp.init.Initializer):
    
    def __init__(self, topology, scale):
      self.topology = topology
      self.scale = scale

    def __call__(self, shape):
        mat = bm.zeros(shape)
        pre_id, post_id = self.topology.cij.requires('pre_ids', 'post_ids')
        #weight =  self.topology.wij
        weight =  np.random.rand(len(pre_id))
        mat[pre_id, post_id] = self.scale * weight 
        #(np.random.randn() * 2 - 1) * 
        return mat


class plane():
    
    def __init__(self, 
                 Ndata: np.ndarray = np.ones((8,8)) / 64,                         # 2D
                 num: int = 1000,               # the number of neurons
                 unit: float = 1,               # Mesh side length
                 connect_len: float = 1,
                 ele_scale: float = 0.25,
                 cell_unit: float = 0.005,
                 cell_scale: list = [1],
                 cell_prob: list = [1],

                 # connect
                 lateral_inh: bool = False,
                 near_p: float = 0.08,
                 far_n: int = 5,
                 messager: list = None,
                 fire_n: int = 10000,

                 # visual
                 draw_3D_face: bool = False,    # 
                 draw_point: bool = False,      # Visual plane
                 draw_connet: bool = False,      # Visual connet
                 savepath: str = "./logs/outcomes/"
                 ):
        
        self.Ndata = Ndata
        self.Nshape = Ndata.shape
        self.Nsize = Ndata.size

        assert self.Ndata.ndim == 2
        assert np.abs(np.sum(self.Ndata) - 1.0) < 1e-9

        self.unit = unit
        self.num = num
        self.connect_len = connect_len
        self.ele_scale = ele_scale 
        self.cell_unit = cell_unit
        self.cluster_size = cell_scale
        self.cluster_prob = np.cumsum(cell_prob)

        self.lateral_inh = lateral_inh
        self.near_p = near_p
        self.far_n = far_n
        self.fire_n = fire_n
        self.messager = messager
        self.savepath = savepath
        self.cluster = []

        isExists=os.path.exists(savepath)
        if not isExists:
            os.makedirs(savepath) 

        if draw_3D_face:
            self._draw_3D_face()

        # (num, 2)  
        # list for each grid
        # [MEA_id, cell_id]  
        self.loc, self.distru, self.input = self._getloc(draw_point)
        self.cij, self.wij = self._getij(draw_connet)
    
    def re_ij(self):
        return self.cij, self.wij

    def _draw_3D_face(self):
        Z = self.Ndata
        plt.pcolormesh(Z.T, cmap="Blues")
        plt.colorbar()
        plt.savefig(os.path.join(self.savepath, "sim_face.png"), dpi = 300)
        plt.cla()
        plt.close()

    def _getloc(self, draw_point):
        '''
        [(x,y,id)]
        '''
        frq = self.Ndata.reshape(-1)
        frq = np.cumsum(frq)
        res = [[] for i in range(frq.size)]
        point = [[],[],[]]
        loc = np.zeros((self.num,3))
        input_id = [[],[]]

        st = []
        for i in range(self.Nshape[0]):
            for j in range(self.Nshape[1]):
                st.append([i * self.unit, j * self.unit])


        num = self.num
        while(num > 0):
            p = np.random.rand()
            id = np.searchsorted(frq, p)
            loc_now =  np.random.rand(2) * self.unit + st[id]
            id_siz = np.searchsorted(self.cluster_prob, np.random.rand())
            cluster = []
            p = int(min(num, self.cluster_size[id_siz]))
            for i in range(p):
                f = self.num - num
                cluster.append(f)
                loc[f,2] = self.cluster_size[id_siz] * self.cell_unit
                loc[f,:2] = loc_now
                num = num - 1
                res[id].append(f)
                point[0].append(loc[f,0])
                point[1].append(loc[f,1])
                point[2].append(loc[f,2])
                dis = np.sqrt(np.sum((loc[f,:2] - st[id] - self.unit / 2)**2))
                if dis <= self.ele_scale + loc[f,2]:
                    input_id[0].append(id)
                    input_id[1].append(f)
            self.cluster.append(cluster)


        if draw_point:
            figure, axes = plt.subplots()
            axes.set_aspect(1)
            axes.set_xlim(0, self.Nshape[0] * self.unit)
            axes.set_ylim(0, self.Nshape[1] * self.unit)
            for i in range(len(st)):
                draw_circle = plt.Circle((st[i][0] + self.unit / 2, st[i][1]+ self.unit / 2), self.ele_scale, color = ele_color)
                axes.add_artist(draw_circle)

            for i in range(len(point[0])):
                draw_circle = plt.Circle((point[0][i], point[1][i]), point[2][i], color = neur_color)
                axes.add_artist(draw_circle)
            
            input = loc[input_id[1]]
            for i in range(input.shape[0]):
                draw_circle = plt.Circle((input[i,0], input[i,1]), input[i,2], color = neur_trans_color)
                axes.add_artist(draw_circle)
            plt.grid(True)
            plt.xticks([])
            plt.yticks([])
            #plt.axis("equal")
            plt.savefig(os.path.join(self.savepath, "sim_points.png"), dpi = 300)
            plt.cla()
            plt.close()

        return loc, res, np.array(input_id)  


    def _getij(self, draw_connet):
        pre_list = []
        post_list = []
        weight = []
        # ============================ cluster ===================================
        for i in range(len(self.cluster)):
            for j in range(len(self.cluster[i])):
                for k in range(j+1, len(self.cluster[i])):
                    p = np.random.rand()
                    if p > 0.5:
                        pre_list.append(self.cluster[i][j])
                        post_list.append(self.cluster[i][k])
                        weight.append(np.random.rand())
                    else:
                        pre_list.append(self.cluster[i][k])
                        post_list.append(self.cluster[i][j])
                        weight.append(np.random.rand())

        # ============================ near connection ===================================
        def connet(x, y):
            for px in self.distru[x]:
                for py in self.distru[y]:
                    if px == py: continue
                    dis = np.sqrt(np.sum((self.loc[px] - self.loc[py]) ** 2))
                    if dis <= self.connect_len and np.random.rand() <= self.near_p:
                        p = np.random.rand()
                        if p < 0.5:  
                            pre_list.append(px)
                            post_list.append(py)
                        else:
                            pre_list.append(py)
                            post_list.append(px)

                        if dis <= self.connect_len / 2:
                            p = 1
                        else: 
                            if self.lateral_inh: p = -1
                        weight.append(p*(self.connect_len - dis))

        for x in range(self.Nshape[0]):
            for y in range(self.Nshape[1]):
                connet(x*self.Nshape[0] + y, x*self.Nshape[0] + y)
                if y+1<self.Nshape[1]:                          connet( x*self.Nshape[0] + y, x*self.Nshape[0] + y + 1)
                if x+1<self.Nshape[0] and y-1>=0:               connet( x*self.Nshape[0] + y, (x+1)*self.Nshape[0] + y - 1)
                if x+1<self.Nshape[0]:                          connet( x*self.Nshape[0] + y, (x+1)*self.Nshape[0] + y)
                if x+1<self.Nshape[0] and y+1<self.Nshape[1]:   connet( x*self.Nshape[0] + y, (x+1)*self.Nshape[0] + y)
        # ============================ near connection ===================================
                
        
        # ============================ far connection ===================================
        for i in range(self.num):
            cnt = self.far_n
            while cnt > 0:
                w = np.random.randint(0, self.num)
                dis = np.sqrt(np.sum((self.loc[w] - self.loc[i]) ** 2))
                if dis > self.connect_len:
                    pre_list.append(w)
                    post_list.append(i)
                    weight.append(np.random.rand())
                    cnt = cnt - 1
        # ============================ far connection ===================================


        # ============================ fire distributed connect ===================================
        if self.messager is not None:
            messager_p = []
            for i in self.messager:
                messager_p = messager_p + self.distru[i]
            
            messager_num = len(messager_p)
            data = self.Ndata.reshape(-1)
            for g in range(self.Nsize):
                here_num = len(self.distru[g])
                if here_num == 0:  continue
                connet_num = self.fire_n * data[g]
                
                while connet_num > 0:
                    x = messager_p[np.random.randint(0, messager_num)]
                    y = self.distru[g][np.random.randint(0, here_num)]
                    pre_list.append(x)
                    post_list.append(y)
                    weight.append(np.random.rand())
                    connet_num = connet_num - 1

        # ============================ fire distributed connect ===================================
        
        pre_list, post_list = np.array(pre_list), np.array(post_list)
        conn = bp.conn.IJConn(i=pre_list, j=post_list)
        conn = conn(pre_size=self.num, post_size=self.num)
        if draw_connet:
            for x,y in zip(pre_list, post_list):
                plt.arrow(self.loc[x,0], self.loc[x,1], self.loc[y,0] - self.loc[x,0], self.loc[y,1] - self.loc[x,1])
            
            plt.savefig(os.path.join(self.savepath, "sim_connect.png"), dpi = 300)
            plt.cla()
            plt.close()

        return conn, np.array(weight)
            



        




