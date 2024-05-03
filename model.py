import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
	def __init__(self, game, device, filters=256, resBlocks=19):
		super().__init__()
		self.device = device

		self.startBlock = nn.Sequential(nn.Conv2d(3, filters, kernel_size=3, padding=1),
										nn.BatchNorm2d(filters),
										nn.ReLU())

		self.resTower = nn.ModuleList([ResBlock(filters) for _ in range(resBlocks)])

		self.policyHead = nn.Sequential(nn.Conv2d(filters, 32, kernel_size=3, padding=1),
										nn.BatchNorm2d(32),
										nn.ReLU(),
										nn.Flatten(),
										nn.Linear(32 * game.rows * game.columns, game.actionSpace))

		self.valueHead = nn.Sequential(nn.Conv2d(filters, 3, kernel_size=3, padding=1),
									   nn.BatchNorm2d(3),
									   nn.ReLU(),
									   nn.Flatten(),
									   nn.Linear(3 * game.rows * game.columns, 1),
									   nn.Tanh())
		self.to(device)

	def forward(self, state):
		state = self.startBlock(state)
		for resBlock in self.resTower:
			state = resBlock(state)
		policy = self.policyHead(state)
		value = self.valueHead(state)
		return policy, value


class ResBlock(nn.Module):
	def __init__(self, filters):
		super().__init__()
		self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
		self.batch1 = nn.BatchNorm2d(filters)
		self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
		self.batch2 = nn.BatchNorm2d(filters)

	def forward(self, state):
		residual = state
		state = F.relu(self.batch1(self.conv1(state)))
		state = self.batch2(self.conv2(state))
		state = state + residual
		state = F.relu(state)
		return state