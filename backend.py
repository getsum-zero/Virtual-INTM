import yaml
from utils.simulator import SNN, defult_setting, getModel, running

if __name__ == '__main__':
    rf = open(file="./config/args.yaml", mode='r')
    crf = rf.read()
    rf.close()
    args = yaml.load(stream=crf, Loader=yaml.FullLoader)
    
    defult_setting(args["Setting"])
    real_data, topology, net = getModel(args)
    running(real_data, topology, net, args)