# encoding=utf-8

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ski_ssim

import torch
import dataloader.dataloaders as dl
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im
import numpy as np
from PIL import Image
from utils import *
from myssim import compare_ssim as ssim
import lpips
from tqdm import tqdm
# from visualize import visualize_grid_attention_v2
import misc_utils as utils
import pdb
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                                save_original_image=False, quality=100):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.01 * img_h, 0.01 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)

def evaluate(model, dataloader, epoch, data_name='val'):
    save_root = os.path.join(opt.result_dir, opt.tag, str(epoch), data_name)

    utils.try_make_dir(save_root)
    input_imgs = []
    # print('Start testing ' + tag + '...')
    for i, sample in enumerate(dataloader):
        utils.progress_bar(i, len(dataloader), 'Eva... ')

        path = sample['path']
        with torch.no_grad():
            recovered = model(sample['input'].to(device=opt.device))
        input_imgs.append(path[0])

        # if data_name == 'val':
        #     recovered = tensor2im(recovered)
        #
        #     save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
        #     Image.fromarray(recovered).save(save_dst)
    return input_imgs

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

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


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
    # pdb.set_trace()
    model = model.to(device=opt.device)
    opt.which_epoch = model.load(opt.load)

    model.eval()
    ############hook part############
    features = []


    ##########这个看的是每个模块最后输出的48通道##############
    def layer_hook(module, input, output):
        features.append(output.data.cpu().squeeze().numpy())
    model.cleaner.net2[4].out_layer.register_forward_hook(layer_hook)


    # ##########这个看的是每个attention模块的输出###############
    # def layer_hook(module,input,output):
    #     features.append(output.data.cpu().squeeze().numpy())
    # model.cleaner.net2[4].att1.register_forward_hook(layer_hook)
    ##########这个看的是attention模块的3个输入###############
    # def layer_hook(module,input):
    #     features.append(input[2].data.cpu().squeeze().numpy())
    # # model.cleaner.net2[4].att3.att.register_forward_hook(layer_hook)  #  attention的hook
    # model.cleaner.net2[4].att3.register_forward_pre_hook(layer_hook)
    ################################

    input_img = evaluate(model, dl.val_dataloader, opt.which_epoch, 'val')
    # pdb.set_trace()
    ################extract attention-map and save############
    for i in tqdm(range(len(features))):
        feature_save_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch), str(i))
        utils.try_make_dir(feature_save_root)
        for j in range(features[i].shape[0]):
            file_name = f'feature_{str(i)}_{str(j)}'
            save_dst = os.path.join(feature_save_root, file_name + '.png')
            img = np.clip(features[i][j],0,1)
            ###########sum################
            img = (img * 255).astype('uint8')
            # pdb.set_trace()
            Image.fromarray(img).save(save_dst)
    ################################################################
    # for i in tqdm(range(len(features))):
    #     feature_save_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch), str(i))
    #     utils.try_make_dir(feature_save_root)
    #     # pdb.set_trace()
    #     for j in range(features[i].shape[0]):
    #         file_name = f'feature_{str(i)}_{str(j)}'
    #         save_dst = os.path.join(feature_save_root, file_name)
    #         img = np.clip(features[i][j],0,1)
    #         visualize_grid_attention_v2(input_img[i],save_path=save_dst,attention_mask=img,save_image=True,quality=100)
    print("Inferenc complete!")



