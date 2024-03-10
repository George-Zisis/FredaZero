import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
	def __init__(self, game, input_size, total_blocks, filter_size, device):
		super().__init__()
		self.device = device

		self.start_block = nn.Sequential(
			nn.Conv2d(input_size, filter_size, kernel_size=3, padding=1),
			nn.BatchNorm2d(filter_size),
			nn.ReLU()
		)

		self.resblocks = nn.ModuleList([ResBlock(filter_size) for i in range(total_blocks)])

		self.policy_head = nn.Sequential(
			nn.Conv2d(filter_size, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(32 * 8 * 8, 1968)
		)

		self.value_head = nn.Sequential(
			nn.Conv2d(filter_size, 3, kernel_size=3, padding=1),
			nn.BatchNorm2d(3),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3 * 8 * 8, 1),
			nn.Tanh()
		)
		self.to(self.device)

	def forward(self, x):
		x = self.start_block(x)
		for resblock in self.resblocks:
			x = resblock(x)
		policy = self.policy_head(x)
		value = self.value_head(x)
		return policy, value


class ResBlock(nn.Module):
	def __init__(self, filter_size):
		super().__init__()
		self.conv1 = nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(filter_size)
		self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(filter_size)

	def forward(self, x):
		residual = x
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))
		x += residual
		x = F.relu(x)
		return x
