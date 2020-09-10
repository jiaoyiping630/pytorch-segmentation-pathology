def main():
    import os
    import pickle
    import argparse
    import yaml
    import numpy as np
    from collections import OrderedDict
    from pathlib import Path
    from tqdm import tqdm
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from utils.preprocess import minmax_normalize, meanstd_normalize

    from models.net import EncoderDecoderNet, SPPNet
    from losses.multi import MultiClassCriterion
    from logger.log import debug_logger
    from logger.plot import history_ploter
    from utils.optimizer import create_optimizer
    from utils.metrics import compute_iou_batch

    gpu_id = 3
    image_folder = r"D:\Projects\MARS-Stomach\Patches\Test"
    save_folder = r"D:\Projects\MARS-Stomach\Patches\Test_predict_yiping"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

    #   构建数据集
    from torch.utils.data import DataLoader, Dataset
    class PredictDataset(Dataset):
        def __init__(self,
                     image_folder,
                     target_size=(512, 512),  # 这应该是输出图像的尺寸
                     net_type='unet',
                     ignore_index=255,  # 忽略的标签值，在VOC里，255表示轮廓线
                     ):

            self.image_folder = Path(image_folder)
            assert net_type in ['unet', 'deeplab']
            self.net_type = net_type
            self.ignore_index = ignore_index

            #   这一系列的操作，都是为了得到图像和label的路径列表
            from pinglib.files import get_file_list_recursive, purename
            image_paths = get_file_list_recursive(image_folder)
            self.image_paths = image_paths

            #   病理图中不要用resize，看看这里有什么需要注意的
            #   可能不需要，因为我们的图像尺寸是统一的！
            if isinstance(target_size, str):
                target_size = eval(target_size)
            self.resizer = None

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            img_path = self.image_paths[index]
            img = np.array(Image.open(img_path))
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))

                img = img.transpose(2, 0, 1)
                img = torch.FloatTensor(img)
            return img

    #   载入模型
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))
    net_config = config['Net']
    train_config = config['Train']
    batch_size = train_config['batch_size']

    dataset = PredictDataset(image_folder=image_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size * 2,
                                shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net_config['output_channels'] = 3
    classes = np.arange(1, 3)

    # Network
    if 'unet' in net_config['dec_type']:
        net_type = 'unet'
        model = EncoderDecoderNet(**net_config)
    else:
        net_type = 'deeplab'
        model = SPPNet(**net_config)

    modelname = config_path.stem
    output_dir = Path('../model') / modelname
    log_dir = Path('../logs') / modelname

    logger = debug_logger(log_dir)
    logger.debug(config)

    # To device
    model = model.to(device)

    # Restore model
    model_path = output_dir.joinpath(f'model_tmp.pth')
    logger.info(f'Resume from {model_path}')
    param = torch.load(model_path)
    model.load_state_dict(param)
    del param

    #   Predict
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader) as _tqdm:
            for batched in _tqdm:
                images, = batched
                images = images.to(device)
                preds = model.tta(images, net_type=net_type)
                preds_np = preds.detach().cpu().numpy()

                labels_np = labels.detach().cpu().numpy()
                iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

                _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
                valid_losses.append(loss.item())
                valid_ious.append(iou)


if __name__ == "__main__":
    main()
