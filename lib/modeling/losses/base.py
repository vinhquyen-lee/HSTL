from ctypes import ArgumentError
import torch.nn as nn
import torch
from utils import Odict
import functools
from utils import ddp_all_gather
from utils.common import safe_get_world_size, is_distributed  # ADD THIS IMPORT


def gather_and_scale_wrapper(func):

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:
            # CHANGED: Only gather if distributed
            if is_distributed():
                for k, v in kwds.items():
                    kwds[k] = ddp_all_gather(v)

            loss, loss_info = func(*args, **kwds)
                    
            # CHANGED: Only gather if distributed
            if is_distributed():
                loss *= safe_get_world_size()
            return loss, loss_info
        # except:
        #     raise ArgumentError
        # # show detailed error traceback
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Loss computation failed: {str(e)}") from e
    return inner


class BaseLoss(nn.Module):

    def __init__(self, loss_term_weight=1.0):

        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight
        self.info = Odict()

    def forward(self, logits, labels):

        return .0, self.info
