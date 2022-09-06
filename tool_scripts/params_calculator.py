from thop import profile
import torch
# from network.Student import Model
from options import opt
from network import get_model
from thop import clever_format
from torchstat import stat
from utils.get_model_summary import get_model_complexity_info



if __name__ == '__main__':
    Model = get_model(opt.model)

    # thop version
    model = Model(opt)
    # model = model.to(device=opt.device)

    if opt.load:
        which_epoch = model.load(opt.load)
    # 使用thop算
    # input = torch.randn(1, 3, 256, 256)
    # input = input.to(device=opt.device)
    # s_macs, s_params = profile(model.cleaner, inputs=(input,))
    # s_macs, s_params = clever_format([s_macs, s_params], "%.3f")


    # print(f'Your model flops is {s_macs}, and params is {s_params}' + '\n')


    #  utils工具版本
    with torch.cuda.device(0):
        model = Model(opt)
    
        # 使用utils里面的工具算，cleaner不要事先放在cuda上
        s_flops, s_params = get_model_complexity_info(model.cleaner, (3, 256, 256), print_per_layer_stat=False)
        print(f'Your model flops is {s_flops}, and params is {s_params}' + '\n')





    # print(f"Teacher network's macs is {t_macs}, and params is {t_params}" + '\n')
    # print(f"Student network's macs is {s_macs}, and params is {s_params}" + '\n')
