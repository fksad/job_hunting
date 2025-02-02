# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/1/26 16:13
# part 1: 导入相关的 package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pydantic import BaseModel

torch.manual_seed(1024)

