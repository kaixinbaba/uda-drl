"""compat with cpu and gpu"""
import torch

use_gpu = torch.cuda.is_available()


def _Tensor(*data, TypeTensor=None):
    if use_gpu:
        return TypeTensor(data).cuda()
    else:
        return TypeTensor(data)


def FloatTensor(*data):
    return _Tensor(*data, TypeTensor=torch.FloatTensor)


def LongTensor(*data):
    return _Tensor(*data, TypeTensor=torch.LongTensor)


def numpy(tensor):
    try:
        return tensor.cpu().detach().numpy()
    except Exception as e:
        print(e)
