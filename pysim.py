import tkinter as tk
from tkinter import filedialog
import numpy as np
import tkinter.font as font
from utils.check import check_covert
from utils.simulator import simulate
import yaml
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from utils.pysimUI import SimUI

'''

    https://ttkbootstrap.readthedocs.io/en/latest/zh/styleguide/entry/
'''

args = {
    "shape":{
        "row": "8",
        "col": "8"
    },
    "real_world_data": {
        "mode": None,
        "stimulus" : None, 
        "response" : None, 
        "draw_ori": False,              
        "draw_p": False, 
        "savepath": "./logs/outcomes/",
        "loadpath": "./logs/",
    },
    "planar_topology":{
        "fire_n": "30000",
        "N": "2000",
        "unit": "1",
        "ele_scale": "0.35",
        "cell_unit": "0.005",
        "cell_scale": "1, 3, 5, 7",
        "cell_prob": "0.85, 0.1, 0.04, 0.01",  

        "connect_len": "0.3",     
        "near_p": "0.0008",         
        "lateral_inh": False,
        "far_n": "2",


        "draw_point": True,      
        "draw_connet": False,   
        "draw_3D_face": False,
        "savepath": "./logs/outcomes/",
    },

    "synapses":{
        "neuron_type": None,
        "synapses_type": None,
        "plasticity_type": None,
        "multiple": "0, 3",                 
        "w_bound": "1",                            
        "homeo_type": "norm",               
        "dy_delete": True,
        "stdp":{
            "tau_s": "40",
            "tau_t": "40",
            "A1": "1",
            "A2": "1",
            "theta_p": "0.001",
            "theta_n": "0.001",
            "method": 'exp_auto',
        },
        "stp":{
            "U": "0.2",
            "tau_d": "2",
            "tau_f": "2",
        },
    },
    

    "Setting":{
        "platform": None,
        "cuda_devices": "0",
        "dt": "0.01"
    },
    "Running":{
        "savepath": "./logs/",
        "cons": "2000",
        "during": "10",
        "epoch": "1",
        "interval": "2", 
    }
}


def bind(event, name, attr):
    val = event.widget
    args[name][attr] = val.get()


class MEAinput():
    def __init__(self, master):

        self.master = master
        self.row = ttk.StringVar(value="8")
        self.col = ttk.StringVar(value="8")
        hdr_txt = "Please enter the size of the electrode array (row, col > 0)" 
        hdr = ttk.Label(master=master, text=hdr_txt, width=50)
        hdr.grid(row=0, column=0, rowspan = 1, columnspan = 2,sticky=NSEW)

        self.entry = ttk.Frame(master)
        self.entry.grid(row=1, column=0, rowspan = 1, columnspan = 1, sticky=NSEW)

        self.button = ttk.Frame(master)
        self.button.grid(row=1, column=1, rowspan = 1, columnspan = 1, sticky=NSEW)

        self.create_form_entry(self.entry, "              Row", self.row)
        self.create_form_entry(self.entry, "              Col", self.col)
        self.create_buttonbox(self.button)

        
    def create_form_entry(self, windows, label, variable):

        def check(x):
            if x.isdigit() and int(x) > 0:
                return True
            else:
                return False
        container = ttk.Frame(windows)
        container.pack(fill=X, expand=YES, pady=5)

        lbl = ttk.Label(master=container, text=label.title(), width=10)
        lbl.pack(side=LEFT, padx=5, anchor='w')

        check_s = container.register(check)
        ent = ttk.Entry(master=container, textvariable=variable, width=8, validate="focus", validatecommand=(check_s, '%P'))
        ent.pack(side=LEFT, padx=5, expand=YES, anchor='w')
    
    def create_buttonbox(self, windows):
        container = ttk.Frame(windows)
        container.pack(fill=X, expand=YES, pady=(15, 10))

        sub_btn = ttk.Button(
            master=container,
            text="Submit",
            command=lambda: self.create_Simulator(args),
            bootstyle=SUCCESS,
            width=10,
        )
        sub_btn.pack(side=LEFT, padx=5)
        sub_btn.focus_set()


    def on_closing(self, window):
        window.destroy()
        self.master.destroy()
        
    
    def create_Simulator(self, args):
        try:
            args["shape"]["row"] = int(self.row.get())
            args["shape"]["col"] = int(self.col.get())
            assert args["shape"]["row"] > 0 
            assert args["shape"]["col"] > 0
        except Exception as e:
            tk.messagebox.showerror(title='Error', message="Please enter correct shape! \n an Integer greater than 0")
            return
        
        try:
            self.master.withdraw()
            simroot = ttk.Toplevel(title="Simulator", resizable=(False, False))

            simwindow = SimUI(simroot, args)
            simwindow.run()
            args = simwindow.args
            print(args)

            simroot.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(simroot))
        except Exception as e:
            self.master.deiconify()
            tk.messagebox.showerror(title='Error', message=str(e))
            # tk.messagebox.showerror(title='Error', message="Unknown Error")
            return


if __name__ == '__main__':
    # font=("Arial",12,"bold")

    pysim = ttk.Window("Initialization", resizable=(False, False))
    MEAinput(pysim)
    pysim.mainloop()




'''
todo:
    检查何处非法
'''    