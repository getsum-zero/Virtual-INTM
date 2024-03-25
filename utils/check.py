import numpy as np


def covert_attr(s):
    if not isinstance(s, str):
        return s
    
    s = s.replace(" ", "") 

    if all(char.isdigit() for char in s):
        return int(s)
    if all(char.isdigit() or char == "." for char in s):
        return float(s)
    if all(char.isdigit() or char == "." or char == "," for char in s):
        return [float(x) for x in s.split(",")]
    return s    

def covert(args):
    if isinstance(args, dict):
        for key in args:
            args[key] = covert(args[key])
    else:
        args = covert_attr(args)
    return args

def check_covert(args):
    return covert(args)