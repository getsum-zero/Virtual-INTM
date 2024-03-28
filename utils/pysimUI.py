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

def list2str(l):
    if isinstance(l,str):  return l
    s = ""
    for i in l:
        s += str(i) + ", "
    return s[:-2]

class SimUI():
    def __init__(self, master, args):

        self.simusRow = int(args["shape"]["row"])
        self.simusCol = int(args["shape"]["col"])
        self.inshape = self.simusRow * self.simusCol

        if args["real_world_data"]["stimulus"] is None:
            args["real_world_data"]["stimulus"] = np.zeros(self.inshape).astype(np.int32).tolist()

        self.args = args
        self.master = master
        self.empty_args = copy.deepcopy(args)
        self.hspacing = 20

    
        style = ttk.Style()
        style.configure('TLabelframe', 
                        font=('Helvetica', 12, 'bold')
                        #bordercolor='blue'
                        )

        # Mode
        option_text = "Select a Mode"
        self.ModeFrame = ttk.Labelframe(self.master, text=option_text, padding=15,)
        self.ModeFrame.grid(row=0, column=0, rowspan = 1, columnspan = 3, sticky=NSEW, pady=5, padx=5)

        # Stimu
        option_text = "Develop stimulus formats"
        self.StimuFrame = ttk.Labelframe(self.master, text=option_text, padding=15, width=10)
        self.StimuFrame.grid(row=1, column=0, rowspan = 2, columnspan = 1, sticky=NSEW, pady=5, padx=5)

        # Core
        option_text = "Core components"
        self.CoreFrame = ttk.Labelframe(self.master, text=option_text, padding=15)
        self.CoreFrame.grid(row=1, column=1, rowspan = 1, columnspan = 2, sticky=NSEW, pady=5, padx=5)

        # Run
        option_text = "Running settings"
        self.RunFrame = ttk.Labelframe(self.master, text=option_text, padding=15)
        self.RunFrame.grid(row=2, column=1, rowspan = 1, columnspan = 2, sticky=NSEW, pady=5, padx=5)

        # Topo
        option_text = "Topology"
        self.TopoFrame = ttk.Labelframe(self.master, text=option_text, padding=15)
        self.TopoFrame.grid(row=3, column=0, rowspan = 1, columnspan = 2, sticky=NSEW, pady=5, padx=5)
        self.CirFrame = ttk.Frame(self.master)
        self.CirFrame.grid(row=3, column=2, rowspan = 1, columnspan = 1, sticky=NSEW, pady=5)

        # defua
        option_text = "Others"
        self.DefuaFrame = ttk.Labelframe(self.master, text=option_text, padding=15)
        self.DefuaFrame.grid(row=4, column=0, rowspan = 1, columnspan = 3, sticky=NSEW, pady=5, padx=5)

        # button
        self.ButtonFrame = ttk.Frame(self.master)
        self.ButtonFrame.grid(row=5, column=0, rowspan = 1, columnspan = 3, sticky=NSEW, pady=5)

        self.init_var()
        self.args2var()

    def init_var(self):
        self.option = tk.IntVar(value=2)
        self.filetext = tk.StringVar(value="Choose the path of real-world data (end with \".mat\")")
        self.configtext = tk.StringVar(value="Choose the path of saved configuration")
        self.var_list = [tk.IntVar(value=0) for _ in range(self.inshape)]

        self.value_n = tk.StringVar(value = "")
        self.value_s = tk.StringVar(value = "")
        self.value_p = tk.StringVar(value = "")
        self.num_sy = tk.StringVar(value = "30000")
        self.pal = tk.StringVar(value = "")
        self.cuda_id = tk.StringVar(value = "0")
        self.cons = tk.StringVar(value = "2000")
        self.T = tk.StringVar(value = "10")
        self.N = tk.StringVar(value = "2000")
        self.unit = tk.StringVar(value = "1")
        self.ele_scale = tk.StringVar(value = "0.35")
        self.scale_var = tk.IntVar(value = int(1000 * 0.005 / 0.35))

        self.canvas = tk.Canvas(self.CirFrame, width=self.hspacing * 10, height=self.hspacing * 11)
        self.canvas.pack()
        r = self.hspacing * 4
        r0 = self.scale_var.get() / 10000 * r
        self.center_x = self.hspacing * 5
        self.center_y = self.hspacing * 4 
        self.canvas.create_oval(self.center_x-r, self.center_y-r, self.center_x+r, self.center_y+r, fill="#7FCBA4", tags="ele")
        self.canvas.create_oval(self.center_x-r0, self.center_y-r0, self.center_x+r0, self.center_y+r0, fill="#4B65AF", tags="cell")
        self.canvas.create_rectangle(self.hspacing, self.hspacing * 9, self.hspacing * 3 , self.hspacing * 9.9, fill="#7FCBA4")
        self.canvas.create_text(self.hspacing * 6, self.hspacing * 9.45, text="electrodes", fill="black")
        self.canvas.create_rectangle(self.hspacing, self.hspacing * 10.1, self.hspacing * 3 , self.hspacing * 11, fill="#4B65AF")
        self.canvas.create_text(self.hspacing * 6, self.hspacing * 10.55, text="cell", fill="black")

        self.cell_scale = tk.StringVar(value= "1, 3, 5, 7")
        self.cell_prob = tk.StringVar(value = "0.85, 0.1, 0.04, 0.01")
        self.connect_len = tk.StringVar(value = "0.3")
        self.near_p = tk.StringVar(value = "0.0008")
        self.lateral_inh = tk.IntVar(value = 0)
        self.far_n = tk.StringVar(value = "2")
        self.multiple = tk.StringVar(value = "0, 3")
        self.w_bound = tk.StringVar(value = "1")
        self.homeo_type = tk.StringVar(value = "norm")
        self.dy_delete = tk.IntVar(value = 1)

        self.tau_s = tk.StringVar(value = "40")
        self.tau_t = tk.StringVar(value = "40")
        self.A1 = tk.StringVar(value = "1")
        self.A2 = tk.StringVar(value = "1")
        self.theta_p = tk.StringVar(value = "0.001")
        self.theta_n = tk.StringVar(value = "0.001")
        self.U = tk.StringVar(value = "0.2")
        self.tau_d = tk.StringVar(value = "2")
        self.tau_f = tk.StringVar(value = "2")
        self.dt = tk.StringVar(value = "0.01")
        self.epoch = tk.StringVar(value = "1")
        self.interval = tk.StringVar(value = "2")

    def args2var(self):
        if self.args["real_world_data"]["mode"] == None:
            self.option.set(2)
            self.filetext.set("Choose the path of real-world data (end with \".mat\")")
            self.configtext.set("Choose the path of saved configuration")
        elif self.args["real_world_data"]["mode"] == "fitting":
            self.option.set(0)
            self.filetext.set(self.args["real_world_data"]["response"])
            self.configtext.set("Choose the path of saved configuration")
        else: 
            self.option.set(1)
            self.configtext.set(self.args["real_world_data"]["loadpath"])
            self.filetext.set("Choose the path of real-world data (end with \".mat\")")

        for i in range(self.inshape):
            self.var_list[i].set(self.args["real_world_data"]["stimulus"][i])
        if self.args["shape"]["row"] > 10 or self.args["shape"]["col"] > 10:
            if self.args["real_world_data"]["stimulus"] == None or np.sum(self.args["real_world_data"]["stimulus"]) == 0:
                self.simcanvas = ttk.Canvas(self.StimuFrame, width=self.hspacing * 14, height=self.hspacing * 14, bg="gray")
                self.simcanvas.grid(row=1, rowspan=10, column=0, columnspan=10)
                self.simcanvas.create_text(self.hspacing * 7, self.hspacing * 7, text="Please load stimulus file !", fill="black")
            else:  self.load_sim(self.args["real_world_data"]["stimulus"])
        
        self.value_n.set('' if self.args["synapses"]["neuron_type"] is None else self.args["synapses"]["neuron_type"])
        self.value_s.set('' if self.args["synapses"]["synapses_type"] is None else self.args["synapses"]["synapses_type"])
        self.value_p.set('' if self.args["synapses"]["plasticity_type"] is None else self.args["synapses"]["plasticity_type"])
        self.num_sy.set(str(self.args["planar_topology"]["fire_n"]))
        self.pal.set('' if self.args["Setting"]["platform"] is None else self.args["Setting"]["platform"])
        self.cuda_id.set(str(self.args["Setting"]["cuda_devices"]))
        self.cons.set(str(self.args["Running"]["cons"]))
        self.T.set(str(self.args["Running"]["during"]))
        self.N.set(str(self.args["planar_topology"]["N"]))
        self.unit.set(str(self.args["planar_topology"]["unit"]))
        self.ele_scale.set(str(self.args["planar_topology"]["ele_scale"]))
        self.scale_var.set(int(float(self.args["planar_topology"]["cell_unit"]) / float(self.args["planar_topology"]["ele_scale"]) * 10000))
        
        self.canvas.delete("cell")
        r = self.scale_var.get() * self.hspacing * 4 / 10000
        self.canvas.create_oval(self.center_x-r, self.center_y-r, 
                                self.center_x+r, self.center_y+r, fill="#4B65AF", tags="cell")

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


    def var2args(self):
        print(self.value_n.get())
        for i in range(self.inshape):
            self.args["real_world_data"]["stimulus"][i] = self.var_list[i].get()
        self.args["synapses"]["neuron_type"] = None if self.value_n.get() == '' else self.value_n.get()
        self.args["synapses"]["synapses_type"] = None if self.value_s.get() == '' else self.value_s.get()
        self.args["synapses"]["plasticity_type"] = None if self.value_p.get() == '' else self.value_p.get()
        self.args["planar_topology"]["fire_n"] = self.num_sy.get()
        self.args["Setting"]["platform"] = None if self.pal.get() == '' else self.pal.get()
        self.args["Setting"]["cuda_devices"] = self.cuda_id.get()
        self.args["Running"]["cons"] = self.cons.get()
        self.args["Running"]["during"] = self.T.get()
        self.args["planar_topology"]["N"] = self.N.get()
        self.args["planar_topology"]["unit"] = self.unit.get()
        self.args["planar_topology"]["ele_scale"] = self.ele_scale.get()
        self.args["planar_topology"]["cell_unit"] = str(self.scale_var.get() / 10000 * float(self.args["planar_topology"]["ele_scale"]))
        
        self.args["planar_topology"]["cell_scale"] = self.cell_scale.get()
        self.args["planar_topology"]["cell_prob"] = self.cell_prob.get()
        self.args["planar_topology"]["connect_len"] = self.connect_len.get()
        self.args["planar_topology"]["near_p"] = self.near_p.get()
        self.args["planar_topology"]["lateral_inh"] = self.lateral_inh.get()
        self.args["planar_topology"]["far_n"] = self.far_n.get()
        self.args["synapses"]["multiple"] = self.multiple.get()
        self.args["synapses"]["w_bound"] = self.w_bound.get()
        self.args["synapses"]["homeo_type"] = self.homeo_type.get()
        self.args["synapses"]["dy_delete"] = self.dy_delete.get()


        self.args["synapses"]["stdp"]["tau_s"] = self.tau_s.get()
        self.args["synapses"]["stdp"]["tau_t"] = self.tau_t.get()
        self.args["synapses"]["stdp"]["A1"] = self.A1.get()
        self.args["synapses"]["stdp"]["A2"] = self.A2.get()
        self.args["synapses"]["stdp"]["theta_p"] = self.theta_p.get()
        self.args["synapses"]["stdp"]["theta_n"] = self.theta_n.get()
        self.args["synapses"]["stp"]["U"] = self.U.get()
        self.args["synapses"]["stp"]["tau_d"] = self.tau_d.get()
        self.args["synapses"]["stp"]["tau_f"] = self.tau_f.get()
        self.args["Setting"]["dt"] = self.dt.get()
        self.args["Running"]["epoch"] = self.epoch.get()
        self.args["Running"]["interval"] = self.interval.get()

    def check_entry(self, x, type):
        if type == "digit":
            return x.isdigit()
        elif type == "float":
            try:  
                float(x) 
                return True
            except ValueError:
                return False
        elif type == "list":
            try:
                [float(y) for y in x.replace(" ", "").split(",")]
                return True
            except:
                return False
        else:
            raise("Unknown check type")
        
    def load_sim(self, sim_list):
        fig = Figure(figsize=(1.85, 1.85), dpi=100)
        ax = fig.add_subplot(111)
        drawpic = np.array(sim_list).reshape(self.simusCol, self.simusCol)
        drawpic = drawpic[-1::-1]
        
        (r, c) = np.where(drawpic > 0)
        r = r + 0.5
        c = c + 0.5
        ax.scatter(c, r, s=30)

        (r, c) = np.where(drawpic == 0)
        r = r + 0.5
        c = c + 0.5
        ax.scatter(c, r, c='none', marker='o',edgecolors='black', s=30)
        
        ax.axis('off')
        fig.tight_layout(pad=0)
        self.simcanvas = FigureCanvasTkAgg(fig, master=self.StimuFrame)
        self.simcanvas.draw()
        self.simcanvas.get_tk_widget().grid(row=1, rowspan=1, column=0, columnspan=1)

    def ModeGenerator(self):
        '''https://blog.csdn.net/Good_Hope/article/details/131133586'''
        def select_file():
            self.args["real_world_data"]["mode"] = "fitting"
            file_path = filedialog.askopenfilename(title="Open", initialdir = "./",
                                    filetypes=[("mat files", "*.mat")])
            print(self.args["real_world_data"]["mode"])
            if file_path:
                self.args["real_world_data"]["response"] = file_path
                self.filetext.set(file_path)
            else:   self.option.set(2)

        def select_folder():
            file_path = filedialog.askdirectory(title ="Select a folder", 
                                                initialdir = "./", mustexist = True)
            try:
                # todo: check the `.pkl`
                if file_path:
                    rf = open(file=os.path.join(file_path, "args.yaml"), mode='r')
                    crf = rf.read()
                    rf.close()
                    self.args = yaml.load(stream=crf, Loader=yaml.FullLoader)
                    self.args["real_world_data"]["mode"] = "simulation"
                    self.args["real_world_data"]["loadpath"] = file_path
                    self.configtext.set(file_path)
                    self.args2var()  # update simus (>10)
                else:    self.option.set(2)
            except Exception as e:
                tk.messagebox.showerror(title='Error', message="Illeagl folder, please select the correct folder!")
                self.option.set(2)
                return
        

        # fitting
        container = ttk.Frame(self.ModeFrame)
        container.pack(fill=X, expand=YES, pady=5)
        self.file_button = ttk.Radiobutton(container, variable=self.option, value=0,
                                    text="Fitting       ", 
                                    bootstyle="success",
                                    command=lambda: select_file())
        self.file_button.pack(side=LEFT, padx=10)
        self.file_label = ttk.Entry(container, textvariable=self.filetext, state="readonly", width=45)
        self.file_label.pack(side=LEFT, padx=5, fill=X, expand=YES,)

        # simulation
        container = ttk.Frame(self.ModeFrame)
        container.pack(fill=X, expand=YES, pady=5)
        self.config_button = ttk.Radiobutton(container, variable=self.option, value=1,
                                    text="Simulation", 
                                    bootstyle="success",
                                    command=lambda: select_folder())
        self.config_button.pack(side=LEFT, padx=10)
        self.config_label = ttk.Entry(container, textvariable=self.configtext, state="readonly", width=45)
        self.config_label.pack(side=LEFT, padx=5, fill=X, expand=YES,)

    def ButtonGenerator(self):

        def run():
            self.var2args()
            print(self.args)
            simulate(check_covert(self.args), self.master)
        def clear():
            self.args = copy.deepcopy(self.empty_args)
            self.args2var()

        def save():
            file_path = filedialog.askdirectory(title ="Select a folder to save", initialdir = "./", mustexist = False)
            try:
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path , "args.yaml"), 'w') as file:
                    yaml.dump(self.args, file)
                # shutil.move(os.path.join(self.args["Running"]["savepath"] , "model.bp"), os.path.join(file_path, "model.bp"))
                shutil.move(os.path.join(self.args["Running"]["savepath"] , "topology.pkl"), os.path.join(file_path, "topology.pkl"))
                tk.messagebox.showinfo(title='', message="Saved successfully")
            except Exception as e:
                tk.messagebox.showerror(title='Error', message=str(e))

        run_button = ttk.Button(master=self.ButtonFrame, text="Run", command=run,
                                bootstyle=PRIMARY, width=10)
        run_button.pack(side=LEFT, expand=True, padx=5)
        run_button.focus_set()

        clear_button = ttk.Button(master=self.ButtonFrame, text="Clear", command=clear,
                                bootstyle=INFO, width=10)
        clear_button.pack(side=LEFT, expand=True, padx=5)
        clear_button.focus_set()

        clear_button = ttk.Button(master=self.ButtonFrame, text="Save", command=save,
                                bootstyle=SUCCESS, width=10)
        clear_button.pack(side=LEFT, expand=True, padx=5)
        clear_button.focus_set()

    def StimuGenerator(self):
        
        def select_file():
            file_path = filedialog.askopenfilename(title="Open", initialdir = "./", filetypes=[("txt files", "*.txt")])
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
                    for i in range(self.inshape):
                        self.var_list[i].set(self.args["real_world_data"]["stimulus"][i])
                except Exception as e:
                    tk.messagebox.showerror(title='Error', message="Illegal txt! \n Make sure the file only contains 0 and 1, and the number of characters is the same as the MEA size!")
                    return    

                if self.mea_type == 0:  self.args2var() 
                else:  self.load_sim(sim_list)


        load = tk.Button(self.StimuFrame, text='load', command=lambda: select_file(), width=8)
        load.grid(row=0, rowspan=1, column=0, columnspan=self.args["shape"]["col"], sticky=W, pady=5)
        
        
        if self.args["shape"]["row"] > 10 or self.args["shape"]["col"] > 10:
            self.mea_type = 1
            self.simcanvas = ttk.Canvas(self.StimuFrame, width=self.hspacing * 14, height=self.hspacing * 14, bg="gray")
            self.simcanvas.grid(row=1, rowspan=10, column=0, columnspan=10)
            self.simcanvas.create_text(self.hspacing * 7, self.hspacing * 7, text="Please load stimulus file !", fill="black")

        else:
            self.mea_type = 0
            for i in range(self.inshape):
                stimulus_button = ttk.Checkbutton(self.StimuFrame, variable=self.var_list[i], 
                                                    onvalue=1, offvalue=0, bootstyle="PRIMARY")
                stimulus_button.grid(row=1 + i//self.simusCol, rowspan=1, column=0 + (i%self.simusCol), columnspan = 1, pady=5,padx=5)


    def adjust_w(self, event):
        val = event.widget.get()
        if val == "BioNMDA" or val == "NMDA":
            self.args["synapses"]["w_bound"] = 10
            self.w_bound.set("10")
            self.master.update()
    
    def CoreGenerator(self):
        
        def create(label, var, values, change = False):
            container = ttk.Frame(self.CoreFrame)
            container.pack(fill=X, expand=YES, pady=5)

            lbl = ttk.Label(master=container, text=label.title(), width=12)
            lbl.pack(side=LEFT, padx=5, anchor='w')
            combobox = ttk.Combobox(master=container, state='readonly', textvariable=var, values=values)
            if change:
                combobox.bind('<<ComboboxSelected>>', lambda event: self.adjust_w(event))
            combobox.pack(side=RIGHT, padx=5, anchor='e')


        create("Neuron type", self.value_n, ['Izhikevich', 'GIF', 'LIF'])
        create("Synapses type", self.value_s, ['AMPA', 'NMDA', 'BioNMDA'], change = True)
        create("Plasticity type", self.value_p, ['STDP',])

        container = ttk.Frame(self.CoreFrame)
        container.pack(fill=X, expand=YES, pady=5)
        lbl = ttk.Label(master=container, text="Synapses number", width=15)
        lbl.pack(side=LEFT, padx=5, anchor='w')

        check_s = container.register(lambda P: self.check_entry(P, "digit"))
        ent = ttk.Entry(master=container, textvariable=self.num_sy, validate="focus", validatecommand=(check_s, '%P'))
        ent.pack(side=RIGHT, padx=5, expand=YES, anchor='e')

    def RunGenerator(self):

        container = ttk.Frame(self.RunFrame)
        container.pack(fill=X, expand=YES, pady=5)
        lbl = ttk.Label(master=container, text="Platform")
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        combobox = ttk.Combobox(master=container, state='readonly', textvariable=self.pal, values=['cpu', 'gpu',], width=8)
        combobox.pack(side=LEFT, padx=5, anchor='w', expand=True)

        lbl = ttk.Label(master=container, text="Cuda id")
        lbl.pack(side=LEFT, padx=5, anchor='e', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "digit"))
        ent = ttk.Entry(master=container, textvariable=self.cuda_id, validate="focus", validatecommand=(check_s, '%P'), width=8)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='e')

        container = ttk.Frame(self.RunFrame)
        container.pack(fill=X, expand=YES, pady=5)
        lbl = ttk.Label(master=container, text="Stimulus intensity")
        lbl.pack(side=LEFT, padx=5, anchor='e', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "float"))
        ent = ttk.Entry(master=container, textvariable=self.cons, validate="focus", validatecommand=(check_s, '%P'), width=8)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')

        lbl = ttk.Label(master=container, text="Time (s)")
        lbl.pack(side=LEFT, padx=5, anchor='e', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "float"))
        ent = ttk.Entry(master=container, textvariable=self.T, validate="focus", validatecommand=(check_s, '%P'), width=8)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='e')

    def TopoGenerator(self):
        def create(text, var, type):
            container = ttk.Frame(self.TopoFrame)
            container.pack(fill=X, expand=YES, pady=5)
            lbl = ttk.Label(master=container, text=text)
            lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)

            check_s = container.register(lambda P: self.check_entry(P, type))
            ent = ttk.Entry(master=container, textvariable=var, validate="focus", validatecommand=(check_s, '%P'), width=20)
            ent.pack(side=LEFT, padx=5, expand=YES, anchor='e')
        
        create("The number of neurons", self.N, "digit")
        create("The distance between electrodes", self.unit, "float")
        create("The size of electrodes", self.ele_scale, "float")



        def update_circle_size(value, canvas, center_x, center_y):
            canvas.delete("cell")
            r = int(float(value)) * self.hspacing * 4 / 10000
            canvas.create_oval(center_x-r, center_y-r, center_x+r, center_y+r, fill="#4B65AF", tags="cell")

        container = ttk.Frame(self.TopoFrame)
        container.pack(fill=X, expand=YES, pady=5)
        lbl = ttk.Label(master=container, text="Relative sizes of cells and electrodes")
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        self.scale = ttk.Scale(container, from_=0, to=10000, orient="horizontal", variable=self.scale_var,
                               command=lambda value: update_circle_size(value, self.canvas, self.center_x, self.center_y, ))
        self.scale.pack(side=LEFT, padx=5, anchor='e', expand=True, fill=X)

    def DefuaGenerator(self):
        
        def creat_Lable_frame(master, strlist, varlist, checklist, widthlist):
            container = ttk.Frame(master)
            container.pack(fill=X, expand=YES, pady=0)
            for text,var,checktype,width in zip(strlist, varlist, checklist,widthlist):
                lbl = ttk.Label(master=container, text=text)
                lbl.pack(side=LEFT, padx=5, anchor='e', expand=True)
                check_s = container.register(lambda P: self.check_entry(P, checktype))
                ent = ttk.Entry(master=container, textvariable=var, validate="focus", 
                                validatecommand=(check_s, '%P'), width=width, justify="center",)
                ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')


        container = ttk.Frame(self.DefuaFrame)
        container.pack(fill=X, expand=YES, pady=5)

        lbl = ttk.Label(master=container, text="Cell scales", width=14)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "list"))
        ent = ttk.Entry(master=container, textvariable=self.cell_scale, validate="focus", 
                        validatecommand=(check_s, '%P'), width=20, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')

        lbl = ttk.Label(master=container, text="dt", width=5)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "float"))
        ent = ttk.Entry(master=container, textvariable=self.dt, validate="focus", 
                        validatecommand=(check_s, '%P'), width=5, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')

        lbl = ttk.Label(master=container, text="Multiple delays",  width=14)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "list"))
        ent = ttk.Entry(master=container, textvariable=self.multiple, validate="focus", 
                        validatecommand=(check_s, '%P'), width=8, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')



        container = ttk.Frame(self.DefuaFrame)
        container.pack(fill=X, expand=YES, pady=5)

        lbl = ttk.Label(master=container, text="Generation prob", width=14)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "list"))
        ent = ttk.Entry(master=container, textvariable=self.cell_prob, validate="focus", 
                        validatecommand=(check_s, '%P'), width=20, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')


        lbl = ttk.Label(master=container, text="Epoch", width=5)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "digit"))
        ent = ttk.Entry(master=container, textvariable=self.epoch, validate="focus", 
                        validatecommand=(check_s, '%P'), width=5, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')

        lbl = ttk.Label(master=container, text="Interval", width=14)
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = container.register(lambda P: self.check_entry(P, "digit"))
        ent = ttk.Entry(master=container, textvariable=self.interval, validate="focus", 
                        validatecommand=(check_s, '%P'), width=8, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')




        # STDP
        option_text = "STDP parameters"
        self.DefuaFrame_STDP = ttk.Labelframe(self.DefuaFrame, text=option_text, padding=15)
        self.DefuaFrame_STDP.pack(fill=X, expand=YES, pady=5)
        strlist = ["tau_s", "tau_t", "A1", "A2", "theta_p", "theta_n"]
        varlist = [self.tau_s, self.tau_t, self.A1, self.A2, self.theta_p, self.theta_n]
        checklist = ["digit", "digit", "float", "float", "float", "float"]
        widthlist = [4,4,3,3,5,5]
        creat_Lable_frame(self.DefuaFrame_STDP, strlist, varlist, checklist, widthlist)

        # STP and Hom
        container = ttk.Frame(self.DefuaFrame)
        container.pack(fill=X, expand=YES, pady=5)
        option_text = "STP parameters"
        self.DefuaFrame_STP = ttk.Labelframe(container, text=option_text, padding=15)
        self.DefuaFrame_STP.pack(pady=5, side=LEFT)
        strlist = ["U", "tau_d", "tau_f"]
        varlist = [self.U, self.tau_d, self.tau_f]
        checklist = ["float", "float", "float"]
        widthlist = [4,4,4]
        creat_Lable_frame(self.DefuaFrame_STP, strlist, varlist, checklist, widthlist)


        option_text = "Homeostasis"
        self.DefuaFrame_Hom = ttk.Labelframe(container, text=option_text, padding=15)
        self.DefuaFrame_Hom.pack(expand=YES, pady=5, side=LEFT, padx=5, fill=X)

        lbl = ttk.Label(master=self.DefuaFrame_Hom, text="Max weight")
        lbl.pack(side=LEFT, padx=5, anchor='w', expand=True)
        check_s = self.DefuaFrame_Hom.register(lambda P: self.check_entry(P, "float"))
        ent = ttk.Entry(master=self.DefuaFrame_Hom, textvariable=self.w_bound, validate="focus", 
                        validatecommand=(check_s, '%P'), width=4, justify="center",)
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')
        label = ttk.Label(self.DefuaFrame_Hom, text="Lateral inh")
        label.pack(side=LEFT, padx=5, anchor='e', expand=True)
        lateral_button = ttk.Checkbutton(self.DefuaFrame_Hom, variable=self.lateral_inh, onvalue=1, offvalue=0, bootstyle="round-toggle")
        lateral_button.pack(side=LEFT, padx=5, expand=YES, anchor='w')
        lbl = ttk.Label(master=self.DefuaFrame_Hom, text="Type")
        lbl.pack(side=LEFT, padx=5, anchor='e', expand=True)
        combobox = ttk.Combobox(master=self.DefuaFrame_Hom, state='readonly', textvariable=self.homeo_type, values=['norm', 'exp',], width=5)
        combobox.pack(side=LEFT, padx=5, anchor='w')


        # Short- and Long- range synaptic
        container = ttk.Frame(self.DefuaFrame)
        container.pack(fill=X, expand=YES, pady=5)

        option_text = "Short-range synaptic"
        self.DefuaFrame_short = ttk.Labelframe(container, text=option_text, padding=15)
        self.DefuaFrame_short.pack(pady=5, side=LEFT)
        strlist = ["Max length", "Connect prob"]
        varlist = [self.connect_len, self.near_p]
        checklist = ["float", "float"]
        widthlist = [8,8]
        creat_Lable_frame(self.DefuaFrame_short, strlist, varlist, checklist, widthlist)

        option_text = "Long-range synaptic"
        self.DefuaFrame_long = ttk.Labelframe(container, text=option_text, padding=15)
        self.DefuaFrame_long.pack(pady=5, side=LEFT, padx = 5)
        strlist = ["Pre neuron"]
        varlist = [self.far_n,]
        checklist = ["digit",]
        widthlist = [4]
        creat_Lable_frame(self.DefuaFrame_long, strlist, varlist, checklist, widthlist)

        option_text = "Structural plasticity"
        self.DefuaFrame_stru = ttk.Labelframe(container, text=option_text, padding=15)
        self.DefuaFrame_stru.pack(expand=YES, pady=5, side=LEFT, padx = 5, fill=X)
        label = ttk.Label(self.DefuaFrame_stru, text="Dynamic del")
        label.pack(side=LEFT, padx=5, anchor='e', expand=True)
        lateral_button = ttk.Checkbutton(self.DefuaFrame_stru, variable=self.dy_delete, onvalue=1, offvalue=0, bootstyle="round-toggle")
        lateral_button.pack(side=LEFT, padx=5, expand=YES, anchor='w')
        

        



    def run(self):
        self.ModeGenerator()
        self.StimuGenerator()
        self.CoreGenerator()
        self.RunGenerator()
        self.TopoGenerator()
        self.DefuaGenerator()
        self.ButtonGenerator()
