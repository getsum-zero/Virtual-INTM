import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
import numpy as np
import tkinter.font as font
from utils.check import check_covert
from utils.simulator import simulate
import yaml
import os
import shutil
import copy
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def draw_line(window, xs, ys, xe, ye):
    canvas = tk.Canvas(window, width=xe-xs, height=ye-ys)
    canvas.place(x = xs , y = ys)
    canvas.create_line(0, (ye-ys)/2, xe-xs, (ye-ys)/2, fill="black")

def list2str(l):
    if isinstance(l,str):  return l
    s = ""
    for i in l:
        s += str(i) + ", "
    return s[:-2]

class SimUI():
    def __init__(self, window, args):
        
        self.window = window
        self.simusRow = int(args["shape"]["row"])
        self.simusCol = int(args["shape"]["col"])
        self.colnum = 30
        self.inshape = self.simusRow * self.simusCol

        self.mea_type = 0

        C = 30
        Cst =13
        Cst2 = 17
        self.hspacing  = 20

        # (row, rowspan, col, colspan)

        self.gird = [
            [(0,1,0,C,'w'), (1,1,1,C-1,'w'), (2,1,1,C-1,'w')],
            [(4,1,0,C,'w'), (5,1,1,1,None), (5,9,1,9,None), (4,1,7,4,'e',8)],
            [(4,1,Cst-1,C-Cst+1,'w'),   # Core  line / 2
             (5,1,Cst,4,'w'), (5,1,Cst2,4,'e',8),
             (6,1,Cst,4,'w'), (6,1,Cst2,4,'e',8),
             (7,1,Cst,4,'w'), (7,1,Cst2,4,'e',8),
             (8,1,Cst,4,'w'), (8,1,Cst2,4,'e',8),  # sy entry
            ],
            [(10,1,Cst-1,C-Cst+1,'w'),  # Running Setting
             (11,1,Cst,4,'w'), (11,1,Cst2,4,'e',8), 
             (12,1,Cst,4,'w'), (12,1,Cst2,4,'e',8), 
             (13,1,Cst,4,'w'), (13,1,Cst2,4,'e',8), 
             (14,1,Cst,4,'w'), (14,1,Cst2,4,'e',8), 
            ],
            [(16,1,1,Cst-4,'w'), (16,1,Cst-3,5,'e',12),
             (17,1,1,Cst-4,'w'), (17,1,Cst-3,5,'e',12),
             (18,1,1,Cst-4,'w'), (18,1,Cst-3,5,'e',12),
             (19,1,1,Cst+1,'w'),
             (20,1,2,Cst,'w'),
             (16,5,Cst+2,9,None),
            ],
            [(21,1,1,15,'w'), (21,1,16,5,'e',25),
             (22,1,1,15,'w'), (22,1,16,5,'e',25),
             (23,1,1,15,'w'), (23,1,16,5,'e',25),
             (24,1,1,15,'w'), (24,1,16,5,'e',25),
             (25,1,1,10,'w'), (25,1,11,4,'e',20), (25,1,15,5,'e'), (25,1,20,1,'e'),
             (26,1,1,10,'w'), (26,1,11,4,'e',20), (26,1,15,5,'e'), (26,1,20,1,'e'),
             (27,1,1,10,'w'), (27,1,11,2,'e',12), (27,1,15,3,'e'), (27,1,18,3,'e', 8),
            ],
            [(28,2,1,5,'w'), 
             (28,1,6,3,'e'), (28,1,9,2,'w',8), (28,1,11,3,'e'), (28,1,14,2,'w',8), (28,1,16,3,'e'), (28,1,19,2,'w',8),
             (29,1,6,3,'e'), (29,1,9,2,'w',8), (29,1,11,3,'e'), (29,1,14,2,'w',8), (29,1,16,3,'e'), (29,1,19,2,'w',8),
             (30,1,1,5,'w'), 
             (30,1,6,3,'e'), (30,1,9,2,'w',8), (30,1,11,3,'e'), (30,1,14,2,'w',8), (30,1,16,3,'e'), (30,1,19,2,'w',8),
             (31,1,1,2,'w'), (31,1,3,3,'w',8), (31,1,8,2,'w'), (31,1,10,3,'w',8), (31,1,15,3,'w'), (31,1,18,3,'w',8),
            ],
            [
                (33,2,0,7), (33,2,7,7), (33,2,14,7), 
            ]
        ]
        self.empty_gird(1,0)        # tab for folder
        self.empty_gird(3,0)        # empty line for folder-to-gird 
        self.empty_gird(5,0)        # empty line for gird-to-core
        self.empty_gird(4,Cst-2)    # tab for core and gird
        self.empty_gird(5,Cst-1)    # tab for core
        self.empty_gird(9,0)        # empty line for core-to-running
        self.empty_gird(11,Cst-1)   # tab for running
        self.empty_gird(15,0)       # empty line for running-to-ele
        self.empty_gird(16,0)       # tab for ele
        self.empty_gird(20,0)       # empty line for ele-to-deafu
        self.empty_gird(32,0)       # empty line for deafu-to-button

        if args["real_world_data"]["stimulus"] is None:
            args["real_world_data"]["stimulus"] = np.zeros(self.inshape).astype(np.int32).tolist()
        self.args = args
        self.empty_args = copy.deepcopy(args)

        self.init_var()
    
    def init_var(self):

        self.option = tk.IntVar()
        self.var_list = [tk.IntVar() for _ in range(self.inshape)]
        self.value_n = tk.StringVar()
        self.value_s = tk.StringVar()
        self.value_p = tk.StringVar()
        self.num_sy = tk.StringVar()
        self.pal = tk.StringVar()
        self.cuda_id = tk.StringVar()
        self.cons = tk.StringVar()
        self.T = tk.StringVar()

        self.N = tk.StringVar()
        self.unit = tk.StringVar()
        self.ele_scale = tk.StringVar()
        self.scale_var = tk.IntVar()

        self.cell_scale = tk.StringVar()
        self.cell_prob = tk.StringVar()
        self.connect_len = tk.StringVar()
        self.near_p = tk.StringVar()
        self.lateral_inh = tk.IntVar()
        self.far_n = tk.StringVar()
        self.multiple = tk.StringVar()
        self.w_bound = tk.StringVar()
        self.homeo_type = tk.StringVar()
        self.dy_delete = tk.IntVar()
        self.tau_s = tk.StringVar()
        self.tau_t = tk.StringVar()
        self.A1 = tk.StringVar()
        self.A2 = tk.StringVar()
        self.theta_p = tk.StringVar()
        self.theta_n = tk.StringVar()
        self.U = tk.StringVar()
        self.tau_d = tk.StringVar()
        self.tau_f = tk.StringVar()
        self.dt = tk.StringVar()
        self.epoch = tk.StringVar()
        self.interval = tk.StringVar()


    def adjust_wbound(self, val):
        if val == "BioNMDA" or val == "NMDA":
            self.args["synapses"]["w_bound"] = 10
            self.w_bound.set("10")
            self.window.update()

    def bind(self, event, name, attr, metr = None, change = None):
        val = event.widget
        if change != None:
            change(val.get())
        if metr:  self.args[name][attr][metr] = val.get()
        else:  self.args[name][attr] = val.get()

    def empty_gird(self, row, col):
        tk.Label(self.window, text="   ").grid(row=row, column=col, columnspan=1, sticky="w") 

    def text_entry(self, window, var, de, text,girdtext, girdentry, name, attr, metr=None):
        var.set(de)
        tk.Label(window, text=text).grid(row=girdtext[0], rowspan=girdtext[1], column=girdtext[2], columnspan=girdtext[3], sticky=girdtext[4])
        entry = tk.Entry(window, textvariable=var, justify="center", width=girdentry[5]) 
        entry.bind("<KeyRelease>", lambda event: self.bind(event, name=name, attr=attr, metr=metr))
        entry.grid(row=girdentry[0], rowspan=girdentry[1], column=girdentry[2], columnspan=girdentry[3], sticky=girdentry[4])

    def update_ui(self):

        print(self.option.get())
        if self.option.get() == 2:
            self.file_button.config(text="Fitting: choose the path of real-world data (end with \".mat\")")
            self.config_button.config(text="Simulation: choose the path of saved configuration")

        for i in range(self.inshape):
            self.var_list[i].set(self.args["real_world_data"]["stimulus"][i])

        if self.mea_type:
            
            if self.args["real_world_data"]["stimulus"] == None or np.sum(self.args["real_world_data"]["stimulus"]) == 0:
                self.simcanvas = tk.Canvas(self.window, width=self.hspacing * 10, height=self.hspacing * 10, bg="gray")
                self.simcanvas.grid(row=self.gird[1][2][0], rowspan=self.gird[1][2][1], 
                                        column=self.gird[1][2][2], columnspan=self.gird[1][2][3])
                self.simcanvas.create_text(self.hspacing * 5, self.hspacing * 5, text="Please load stimulus file !", fill="black")
            else:
                self.load_sim(self.args["real_world_data"]["stimulus"])

        self.value_n.set(self.args["synapses"]["neuron_type"])
        self.value_s.set(self.args["synapses"]["synapses_type"])
        self.value_p.set(self.args["synapses"]["plasticity_type"])
        self.num_sy.set(str(self.args["planar_topology"]["fire_n"]))
        self.pal.set(self.args["Setting"]["platform"])
        self.cuda_id.set(str(self.args["Setting"]["cuda_devices"]))
        self.cons.set(str(self.args["Running"]["cons"]))
        self.T.set(str(self.args["Running"]["during"]))
        self.N.set(str(self.args["planar_topology"]["N"]))
        self.unit.set(str(self.args["planar_topology"]["unit"]))
        self.ele_scale.set(str(self.args["planar_topology"]["ele_scale"]))
        
        self.scale.set(int(float(self.args["planar_topology"]["cell_unit"]) / float(self.args["planar_topology"]["ele_scale"]) * 10000))
        self.scale_var.set(int(float(self.args["planar_topology"]["cell_unit"]) / float(self.args["planar_topology"]["ele_scale"]) * 10000))
        self.canvas.delete("cell")
        r =  self.scale_var.get() * self.hspacing * 2.5 / 10000
        self.canvas.create_oval(self.center_x-r, self.center_y-r, self.center_x+r, self.center_y+r, fill="#4B65AF", tags="cell")

        self.cell_scale.set(list2str(self.args["planar_topology"]["cell_scale"]))
        self.cell_prob.set(list2str(self.args["planar_topology"]["cell_prob"]))
        self.connect_len.set(str(self.args["planar_topology"]["connect_len"]))
        self.near_p.set(str(self.args["planar_topology"]["near_p"]))
        self.lateral_inh.set(self.args["planar_topology"]["lateral_inh"])
        self.far_n.set(str(self.args["planar_topology"]["far_n"]))
        self.multiple.set(list2str(self.args["synapses"]["multiple"]))
        self.w_bound.set(str(self.args["synapses"]["w_bound"]))
        self.homeo_type.set(self.args["synapses"]["homeo_type"])
        self.dy_delete.set(self.args["synapses"]["dy_delete"])
        self.tau_s.set(str(self.args["synapses"]["stdp"]["tau_s"]))
        self.tau_t.set(str(self.args["synapses"]["stdp"]["tau_t"]))
        self.A1.set(str(self.args["synapses"]["stdp"]["A1"]))
        self.A2.set(str(self.args["synapses"]["stdp"]["A2"]))
        self.theta_p.set(str(self.args["synapses"]["stdp"]["theta_p"]))
        self.theta_n.set(str(self.args["synapses"]["stdp"]["theta_n"]))
        self.U.set(str(self.args["synapses"]["stp"]["U"]))
        self.tau_d.set(str(self.args["synapses"]["stp"]["tau_d"]))
        self.tau_f.set(str(self.args["synapses"]["stp"]["tau_f"]))
        self.dt.set(str(self.args["Setting"]["dt"]))
        self.epoch.set(str(self.args["Running"]["epoch"]))
        self.interval.set(str(self.args["Running"]["interval"]))
        
        self.window.update()

    def select_file_or_config(self):
        window = self.window
        hspacing = self.hspacing
        '''
            https://blog.csdn.net/Good_Hope/article/details/131133586
        '''
        def select_file(button):
            self.args["real_world_data"]["mode"] = "fitting"
            file_path = filedialog.askopenfilename(title="Open", initialdir = "./",
                                    filetypes=[("mat files", "*.mat")])
            if file_path:
                self.args["real_world_data"]["response"] = file_path
                button.config(text = "Fitting:" + file_path)
            else:
                self.option.set(2)

        def select_folder(button):
            file_path = filedialog.askdirectory(title ="Select a folder", 
                                                initialdir = "./", mustexist = True)
            
            try:
                if file_path:
                    rf = open(file=os.path.join(file_path, "args.yaml"), mode='r')
                    crf = rf.read()
                    rf.close()
                    self.args = yaml.load(stream=crf, Loader=yaml.FullLoader)
                    self.args["real_world_data"]["mode"] = "simulation"
                    self.args["real_world_data"]["loadpath"] = file_path
                    button.config(text = "Simulation:" + file_path)
                    self.update_ui()
                else:
                    self.option.set(2)
            except Exception as e:
                tk.messagebox.showerror(title='Error', message="Illeagl folder, please select the correct folder!")
                self.option.set(2)
                return

        self.option.set(2)
        label = tk.Label(window, text="Select a Mode:")
        gird = self.gird[0]
        label.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3], sticky=gird[0][4])

        self.file_button = tk.Radiobutton(window, variable=self.option, value=0,
                                    text="Fitting: choose the path of real-world data (end with \".mat\")", 
                                    command=lambda: select_file(self.file_button))
        self.file_button.grid(row=gird[1][0], rowspan=gird[1][1], column=gird[1][2], columnspan=gird[1][3], sticky=gird[1][4])

        self.config_button = tk.Radiobutton(window, variable=self.option, value=1,
                                    text="Simulation: choose the path of saved configuration", 
                                    command=lambda: select_folder(self.config_button))
        self.config_button.grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3], sticky=gird[2][4])

    def load_sim(self, sim_list):
        gird = self.gird[1]
        fig = Figure(figsize=(1.85, 1.85), dpi=100)
        ax = fig.add_subplot(111)
        drawpic = np.array(sim_list).reshape(self.simusCol, self.simusCol)
        drawpic = drawpic[-1::-1]
        
        (r, c) = np.where(drawpic > 0)
        r = r + 0.5
        c = c + 0.5
        ax.scatter(c, r, s=16)

        (r, c) = np.where(drawpic == 0)
        r = r + 0.5
        c = c + 0.5
        ax.scatter(c, r, s=16, c='none', marker='o',edgecolors='black')
        

        #ax.pcolormesh(drawpic)
        ax.axis('off')
        fig.tight_layout(pad=0)
        self.simcanvas = FigureCanvasTkAgg(fig, master=self.window)
        self.simcanvas.draw()
        self.simcanvas.get_tk_widget().grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3])
   
    def stimulus_grid(self):

        def select_file(type):
            file_path = filedialog.askopenfilename(title="Open", initialdir = "./",
                                    filetypes=[("txt files", "*.txt")])
            if file_path:
                try:
                    file = open(file_path, 'r')
                    content = file.readlines()
                    file.close()
                except Exception as e:
                    tk.messagebox.showerror(title='Error', message="Failed to open txt, please select the correct file!")
                    return
                
                try:
                    sim_list = []
                    for line in content:
                        for val in line.strip():
                            v = int(val)
                            assert v!=0 or v!=1
                            sim_list.append(v)
                    assert len(sim_list) == self.inshape
                    self.args["real_world_data"]["stimulus"] = sim_list
                except Exception as e:
                    tk.messagebox.showerror(title='Error', message="Illegal txt! \n Make sure the file only contains 0 and 1, and the number of characters is the same as the MEA size!")
                    return    

                if self.mea_type == 0:
                    self.update_ui() 
                else:
                    self.load_sim(sim_list)

               

        window = self.window
        gird = self.gird[1]

        label = tk.Label(window, text="Develop stimulus formats:")
        label.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3], sticky=gird[0][4])
        load = tk.Button(window, text='load', 
                            width=gird[3][5], command=lambda: select_file(type))
        load.grid(row=gird[3][0], rowspan=gird[3][1], column=gird[3][2], columnspan=gird[3][3], sticky=gird[3][4])
        
        
        if self.args["shape"]["row"] > 10 or self.args["shape"]["col"] > 10:
            self.mea_type = 1
            self.simcanvas = tk.Canvas(window, width=self.hspacing * 10, height=self.hspacing * 10, bg="gray")
            self.simcanvas.grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3])
            self.simcanvas.create_text(self.hspacing * 5, self.hspacing * 5, text="Please load stimulus file !", fill="black")

        
        else:
            self.mea_type = 0
            def stimulus_array_func():
                for i in range(self.inshape):
                    self.args["real_world_data"]["stimulus"][i] = self.var_list[i].get()
    
            for i in range(self.inshape):
                self.var_list[i].set(self.args["real_world_data"]["stimulus"][i])
                stimulus_button = tk.Checkbutton(window, variable=self.var_list[i], 
                                                    onvalue=1, offvalue=0, bd=0,
                                                    command=lambda: stimulus_array_func())
                stimulus_button.grid(row=gird[1][0] + i//self.simusCol, rowspan=gird[1][1], column= gird[1][2] + (i%self.simusCol), columnspan = gird[1][3])

    def set_button(self):
        window = self.window
        gird = self.gird[7]

        def run(window):
            simulate(check_covert(self.args), window)
        def clear():
            self.option.set(2)
            self.args = copy.deepcopy(self.empty_args)
            self.update_ui()

        def save():
            file_path = filedialog.askdirectory(title ="Select a folder to save", initialdir = "./", mustexist = False)
            try:
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path , "args.yaml"), 'w') as file:
                    yaml.dump(self.args, file)
                shutil.move(os.path.join(self.args["Running"]["savepath"] , "model.bp"), os.path.join(file_path, "model.bp"))
                shutil.move(os.path.join(self.args["Running"]["savepath"] , "topology.pkl"), os.path.join(file_path, "topology.pkl"))
                tk.messagebox.showinfo(title='', message="Saved successfully")
            except Exception as e:
                tk.messagebox.showerror(title='Error', message=str(e))
        run_button = tk.Button(window, text='Run', font=('Arial', 12), 
                            width=10, height=1, command=lambda: run(window))
        run_button.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3])
        run_button = tk.Button(window, text='Clear', font=('Arial', 12), 
                            width=10, height=1, command=clear)
        run_button.grid(row=gird[1][0], rowspan=gird[1][1], column=gird[1][2], columnspan=gird[1][3])
        run_button = tk.Button(window, text='Save', font=('Arial', 12), 
                            width=10, height=1, command=save)
        run_button.grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3])

    def Key_para(self):
        window = self.window
        hspacing = self.hspacing
        '''
            https://www.cnblogs.com/rainbow-tan/p/14134820.html
        '''

        gird = self.gird[2]
        # 1 line
        label = tk.Label(window, text="Core components:")
        label.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3], sticky=gird[0][4])


        # last 1 line 
        # neuron type  1 line
        label = tk.Label(window, text="Neuron type:")
        label.grid(row=gird[1][0], rowspan=gird[1][1], column=gird[1][2], columnspan=gird[1][3], sticky=gird[1][4])
        values_n = ['Izhikevich', 'GIF', 'LIF']
        combobox = ttk.Combobox(master=window, state='readonly', textvariable=self.value_n, values=values_n, width=gird[2][5])
        combobox.bind('<<ComboboxSelected>>', lambda event: self.bind(event, name="synapses", attr="neuron_type"))
        combobox.grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3], sticky=gird[2][4])

        # synapses type
        label = tk.Label(window, text="Synapses type:")
        label.grid(row=gird[3][0], rowspan=gird[3][1], column=gird[3][2], columnspan=gird[3][3], sticky=gird[3][4])
        values_s = ['AMPA', 'NMDA', 'BioNMDA']
        combobox = ttk.Combobox(master=window, state='readonly', textvariable=self.value_s, values=values_s, width=gird[4][5])
        combobox.bind('<<ComboboxSelected>>', lambda event: self.bind(event, name="synapses", attr="synapses_type", change=self.adjust_wbound))
        combobox.grid(row=gird[4][0], rowspan=gird[4][1], column=gird[4][2], columnspan=gird[4][3], sticky=gird[4][4])

        # plasticity type
        label = tk.Label(window, text="Plasticity type:")
        label.grid(row=gird[5][0], rowspan=gird[5][1], column=gird[5][2], columnspan=gird[5][3], sticky=gird[5][4])
        values_p = ['STDP',]
        combobox = ttk.Combobox(master=window, state='readonly', textvariable=self.value_p, values=values_p, width=gird[6][5])
        combobox.bind('<<ComboboxSelected>>', lambda event: self.bind(event, name="synapses", attr="plasticity_type"))
        combobox.grid(row=gird[6][0], rowspan=gird[6][1], column=gird[6][2], columnspan=gird[6][3], sticky=gird[6][4])

        self.text_entry(window, self.num_sy, self.args["planar_topology"]["fire_n"], "\"Distributed\" synapses: ", 
                gird[7], gird[8], "planar_topology", "fire_n")

        # ======= Running setting ========
        gird = self.gird[3]
        label = tk.Label(window, text="Running settings:")
        label.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3], sticky=gird[0][4])

        label = tk.Label(window, text="Platform:")
        label.grid(row=gird[1][0], rowspan=gird[1][1], column=gird[1][2], columnspan=gird[1][3], sticky=gird[1][4])
        values = ['cpu', 'gpu',]
        combobox = ttk.Combobox(master=window, state='readonly', textvariable=self.pal, values=values, width=gird[2][5])
        combobox.bind('<<ComboboxSelected>>', lambda event: self.bind(event, name="Setting", attr="platform"))
        combobox.grid(row=gird[2][0], rowspan=gird[2][1], column=gird[2][2], columnspan=gird[2][3], sticky=gird[2][4])

        # cuda _id
        self.text_entry(window, self.cuda_id, self.args["Setting"]["cuda_devices"], "Cuda id: ", 
                gird[3], gird[4], "Setting", "cuda_devices")
        self.text_entry(window, self.cons, self.args["Running"]["cons"], "Stimulus intensity: ", 
                gird[5], gird[6], "Running", "cons")
        self.text_entry(window, self.T, self.args["Running"]["during"], "T (s) : ", 
                 gird[7], gird[8], "Running", "during")

    def topology_para(self):
        window = self.window
        hspacing = self.hspacing

        gird = self.gird[4]

        self.text_entry(window, self.N, "2000", "The number of neurons: ",
                gird[0], gird[1],"planar_topology", "N")
        self.text_entry(window, self.unit, "1", "The distance between electrodes: ", 
                gird[2], gird[3], "planar_topology", "unit")
        self.text_entry(window, self.ele_scale, "0.35", "The size of electrodes: ", 
                gird[4], gird[5], "planar_topology", "ele_scale")
        
        def update_circle_size(value, canvas, center_x, center_y):
            canvas.delete("cell")
            self.args["planar_topology"]["cell_unit"] = str(float(self.args["planar_topology"]["ele_scale"]) * int(value) / 10000)
            r = int(value) * hspacing * 2.5 / 10000
            canvas.create_oval(center_x-r, center_y-r, center_x+r, center_y+r, fill="#4B65AF", tags="cell")

        self.canvas = tk.Canvas(window, width=hspacing * 7, height=hspacing * 7)
        self.canvas.grid(row=gird[8][0], rowspan=gird[8][1], column=gird[8][2], columnspan=gird[8][3], sticky=gird[8][4])
        r = hspacing * 2.5
        r0 = float(self.args["planar_topology"]["cell_unit"]) / float(self.args["planar_topology"]["ele_scale"]) * r
        self.center_x = hspacing * 3.5
        self.center_y = hspacing * 2.5 
        self.canvas.create_oval(self.center_x-r, self.center_y-r, self.center_x+r, self.center_y+r, fill="#7FCBA4", tags="ele")
        self.canvas.create_oval(self.center_x-r0, self.center_y-r0, self.center_x+r0, self.center_y+r0, fill="#4B65AF", tags="cell")
        self.canvas.create_rectangle(hspacing, hspacing * 5.25, hspacing * 2 , hspacing * 5.75, fill="#7FCBA4")
        self.canvas.create_text(hspacing * 4, hspacing * 5.5, text="electrodes", fill="black")
        self.canvas.create_rectangle(hspacing, hspacing * 6.25, hspacing * 2 , hspacing * 6.75, fill="#4B65AF")
        self.canvas.create_text(hspacing * 3.15, hspacing * 6.5, text="cell", fill="black")

        tk.Label(window, text="Relative sizes of cells and electrodes: ").grid(row=gird[6][0], rowspan=gird[6][1], column=gird[6][2], 
                                                                               columnspan=gird[6][3], sticky=gird[6][4])
        self.scale_var.set(int(r0 / r * 10000))
        self.scale = tk.Scale(window, from_=0, to=10000, orient="horizontal", length= hspacing * 10, variable=self.scale_var,
                        command=lambda value: update_circle_size(value, self.canvas, self.center_x, self.center_y, ))
        self.scale.grid(row=gird[7][0], rowspan=gird[7][1], column=gird[7][2], columnspan=gird[7][3], sticky=gird[7][4])

    def default_para(self):
        window = self.window
        gird = self.gird[5]
        # draw_line(window, 0, hspacing * 19.25, hspacing * 24, hspacing * 19.5)
        self.text_entry(window, self.cell_scale, "1, 3, 5, 7", "The list of cell scales: ", 
                gird[0], gird[1], "planar_topology", "cell_scale")
        self.text_entry(window, self.cell_prob, "0.5, 0.3, 0.15, 0.05", "Generation probability of cells: ", 
                gird[2], gird[3], "planar_topology", "cell_prob")          
        self.text_entry(window, self.connect_len, "0.3", "Maximum connection length of short-range synapses: ", 
                gird[4], gird[5], "planar_topology", "connect_len")
        self.text_entry(window, self.near_p,"0.0008", "The connection probability of short-range synaptic: ", 
                gird[6], gird[7], "planar_topology", "near_p")
        
        self.text_entry(window, self.far_n, "2", "Long-range synaptic pre neuron: ", 
                gird[8], gird[9], "planar_topology", "far_n")
        def check_func(var, name, attr):
            self.args[name][attr] = var.get()
        label = tk.Label(window, text="Lateral inhibition: ")
        label.grid(row=gird[10][0], rowspan=gird[10][1], column=gird[10][2], columnspan=gird[10][3], sticky=gird[10][4])
        self.lateral_inh.set(0)
        lateral_button = tk.Checkbutton(window, variable=self.lateral_inh, onvalue=1, offvalue=0, 
                                        command=lambda: check_func(self.lateral_inh, "planar_topology", "lateral_inh"))
        lateral_button.grid(row=gird[11][0], rowspan=gird[11][1], column=gird[11][2], columnspan=gird[11][3], sticky=gird[11][4])


                                    
        self.text_entry(window, self.multiple, "0, 3", "Multiple synaptic delays: ", 
                gird[12], gird[13], "synapses", "multiple")
        label = tk.Label(window, text="Dynamic deletion: ")
        label.grid(row=gird[14][0], rowspan=gird[14][1], column=gird[14][2], columnspan=gird[14][3], sticky=gird[14][4]) 
        self.dy_delete.set(0)
        dy_delete_button = tk.Checkbutton(window, variable=self.dy_delete, onvalue=1, offvalue=0, 
                                        command=lambda: check_func(self.dy_delete, "synapses", "dy_delete"))
        dy_delete_button.grid(row=gird[15][0], rowspan=gird[15][1], column=gird[15][2], columnspan=gird[15][3], sticky=gird[15][4])

        self.text_entry(window, self.w_bound, "1", "Maximum synaptic weight: ", 
                gird[16], gird[17],"synapses", "w_bound",)
        label = tk.Label(window, text="Homeostasis type:  ")
        label.grid(row=gird[18][0], rowspan=gird[18][1], column=gird[18][2], columnspan=gird[18][3], sticky=gird[18][4])

        values = ['norm', 'exp',]
        self.homeo_type.set('norm')
        combobox = ttk.Combobox(master=window, state='readonly', textvariable=self.homeo_type, values=values, width=gird[19][5])
        combobox.current(0)
        combobox.bind('<<ComboboxSelected>>', lambda event: self.bind(event, name="synapses", attr="homeo_type"))
        combobox.grid(row=gird[19][0], rowspan=gird[19][1], column=gird[19][2], columnspan=gird[19][3], sticky=gird[19][4])


        gird = self.gird[6]
        label = tk.Label(window, text="STDP parameters:")
        label.grid(row=gird[0][0], rowspan=gird[0][1], column=gird[0][2], columnspan=gird[0][3], sticky=gird[0][4])
        self.text_entry(window, self.tau_s, "40", "tau_s: ", gird[1], gird[2],"synapses", "stdp", metr="tau_s")
        self.text_entry(window, self.tau_t, "40", "tau_t: ", gird[3], gird[4],"synapses", "stdp", metr="tau_t")
        self.text_entry(window, self.A1, "1", "A1: ",gird[5], gird[6], "synapses", "stdp", metr="A1")
        
        self.text_entry(window, self.A2, "1", "A2: ", gird[7], gird[8], "synapses", "stdp", metr="A2")
        self.text_entry(window, self.theta_p, "0.001", "theta_p: ", gird[9], gird[10], "synapses", "stdp", metr="theta_p")
        self.text_entry(window, self.theta_n, "0.001", "theta_n: ", gird[11], gird[12],"synapses", "stdp", metr="theta_n")
        
        label = tk.Label(window, text="STD parameters:")
        label.grid(row=gird[13][0], rowspan=gird[13][1], column=gird[13][2], columnspan=gird[13][3], sticky=gird[13][4])
        self.text_entry(window, self.U, "0.2", "U: ", gird[14], gird[15], "synapses", "stp", metr="U")
        self.text_entry(window, self.tau_d, "2", "tau_d: ", gird[16], gird[17], "synapses", "stp", metr="tau_d")
        self.text_entry(window, self.tau_f, "2", "tau_f: ", gird[18], gird[19], "synapses", "stp",metr="tau_f")
        
        self.text_entry(window, self.dt, "0.01", "dt: ", gird[20], gird[21], "Setting", "dt")
        self.text_entry(window, self.epoch, "1", "epoch: ", gird[22], gird[23], "Running", "epoch")
        self.text_entry(window, self.interval, "2", "interval: ", gird[24], gird[25], "Running", "interval")

    def show(self):
        self.select_file_or_config()
        self.stimulus_grid()
        self.Key_para()
        self.topology_para()
        self.default_para()
        self.set_button()
        
        
        
    