import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import tkinter.font as font
from utils.check import check_covert
from utils.simulator import simulate
import yaml
import os

from utils.draw import SimUI

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

def create_Simulator(old, args):
    try:
        args["shape"]["row"] = int(args["shape"]["row"])
        args["shape"]["col"] = int(args["shape"]["col"])
        print(args["shape"])
        assert args["shape"]["row"] > 0 
        assert args["shape"]["col"] > 0
    except Exception as e:
        tk.messagebox.showerror(title='Error', message="Please enter correct shape! \n an Integer greater than 0")
        return
    
    old.destroy()
    root = tk.Tk()
    root.title('Simulator')
    root.geometry('650x850')

    window = SimUI(root, args)
    window.show()
    root.deiconify()  # 显示窗口
    root.mainloop()
    args = window.args
    print(args)


def input_shape():
    root = tk.Tk()
    root.title("Initialization")
    root.geometry('380x80')
    root.resizable(0,0)
    tk.Label(root, text="Please enter the size of the electrode array ( row, col > 0 ): ", ).grid(
        row=0, rowspan=1, column=0, columnspan=3)
    
    row = tk.StringVar()
    col = tk.StringVar()
    row.set("8")
    col.set("8")

    tk.Label(root, text="Row:").grid(row=1, rowspan=1, column=0, columnspan=1)
    entry = tk.Entry(root, textvariable=row, justify="center", width=10) 
    entry.bind("<KeyRelease>", lambda event: bind(event, name="shape", attr="row"))
    entry.grid(row=1, rowspan=1, column=1, columnspan=1, sticky='w')

    
    tk.Label(root, text="Col:").grid(row=2, rowspan=1, column=0, columnspan=1)
    entry = tk.Entry(root, textvariable=col, justify="center", width=10) 
    entry.bind("<KeyRelease>", lambda event: bind(event, name="shape", attr="col"))
    entry.grid(row=2, rowspan=1, column=1, columnspan=1, sticky='w')

    run_button = tk.Button(root, text='Submit', 
                            width=10, height=1, command=lambda: create_Simulator(root, args))
    run_button.grid(row=1, rowspan=2, column=2, columnspan=1)

    root.mainloop()


if __name__ == '__main__':
    # font=("Arial",12,"bold")
    input_shape()



'''
todo:
    检查何处非法
'''    