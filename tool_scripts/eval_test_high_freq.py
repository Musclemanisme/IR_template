# encoding=utf-8

# from skimage.measure import compare_psnr as psnr
# from skimage.measure import compare_ssim as ski_ssim  # deprecated
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ski_ssim

import torch
import torch.nn as nn
import dataloader.dataloaders as dl
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im

from PIL import Image
from utils import *

import misc_utils as utils
import pdb


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_LL,x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h




class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):

    save_root = os.path.join(opt.result_dir, opt.tag, str(epoch), data_name)

    utils.try_make_dir(save_root)

    total_psnr = 0.0
    total_ssim = 0.0
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, sample in enumerate(dataloader):
        utils.progress_bar(i, len(dataloader), 'Eva... ')

        path = sample['path']
        with torch.no_grad():
            recovered, l1, l2, l3 = model(sample['input'].to(device=opt.device))
            l1_h = l1[:, 0:3, :, :]
            l1_v = l1[:, 3:6, :, :]
            l1_dia = l1[:, 6:9, :, :]
            l2_h = l2[:, 0:3, :, :]
            l2_v = l2[:, 3:6, :, :]
            l2_dia = l2[:, 6:9, :, :]
            l3_h = l3[:, 0:3, :, :]
            l3_v = l3[:, 3:6, :, :]
            l3_dia = l3[:, 6:9, :, :]
            input_list = [l1_h, l1_v, l1_dia, l2_h, l2_v, l2_dia, l3_h, l3_v, l3_dia]
            # for item in input_list:
            #     item = tensor2im(item)
            #     item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        if data_name == 'val':
            label = sample['label'].permute(0,3,1,2)
            l2_label = DWT()(label)[0]
            l3_label = DWT()(l2_label)[0]

            gt_l1_h = DWT()(label)[1][:, 3:6, :, :]
            gt_l1_v = DWT()(label)[1][:, 6:9, :, :]
            gt_l1_dia = DWT()(label)[1][:, 9:12, :, :]
            gt_l2_h = DWT()(l2_label)[1][:, 3:6, :, :]
            gt_l2_v = DWT()(l2_label)[1][:, 6:9, :, :]
            gt_l2_dia = DWT()(l2_label)[1][:, 9:12, :, :]
            gt_l3_h = DWT()(l3_label)[1][:, 3:6, :, :]
            gt_l3_v = DWT()(l3_label)[1][:, 6:9, :, :]
            gt_l3_dia = DWT()(l3_label)[1][:, 9:12, :, :]
            label_list = [gt_l1_h, gt_l1_v, gt_l1_dia, gt_l2_h, gt_l2_v, gt_l2_dia, gt_l3_h, gt_l3_v, gt_l3_dia]
            # for item in label_list:
            #     item = tensor2im(item)
            #     item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
            final_list = []
            # for i in range(len(input_list)):
            #     print(str(i),':','input_shape:',input_list[i].shape,'label_shape:',label_list[i].shape)
            for i in range(len(input_list)):
                input_ = input_list[i]
                input_ = tensor2im(input_)
                input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2BGR)
                label_ = label_list[i]
                label_ = tensor2im(label_)
                label_ = cv2.cvtColor(label_, cv2.COLOR_RGB2BGR)
                # print('input_shape:', input_.shape, 'label_shape:', label_.shape)
                # import pdb
                # pdb.set_trace()
                output = np.concatenate((input_, label_), axis=1)
                dst = os.path.join(save_root, utils.get_file_name(path[0]) +f'_{i}' + '.png')

                # import pdb
                # pdb.set_trace()
                cv2.imwrite(dst, output)



            # save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
            # Image.fromarray(recovered).save(save_dst)

        elif data_name == 'test':
            pass

        else:
            raise Exception('Unknown dataset name: %s.' % data_name)

        # 保存结果
        # save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
        # Image.fromarray(recovered).save(save_dst)

    # if data_name == 'val':
    #     ave_psnr = total_psnr / float(ct_num)
    #     ave_ssim = total_ssim / float(ct_num)
    #     # write_loss(writer, f'val/{data_name}', 'psnr', total_psnr / float(ct_num), epochs)
    #
    #     logger.info(f'Eva({data_name}) epoch {epoch}, psnr: {ave_psnr}.')
    #     logger.info(f'Eva({data_name}) epoch {epoch}, ssim: {ave_ssim}.')
    #
    #     return f'{ave_ssim: .3f}'
    # else:
    #     return ''


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    opt.which_epoch = model.load(opt.load)

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)
    evaluate(model, dl.val_dataloader, opt.which_epoch, writer, logger, 'val')

