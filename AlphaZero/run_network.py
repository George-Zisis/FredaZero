import numpy as np

import state_mod
import chess
import torch

from agent import AlphaZero
from chess_env import ChessEnv
from network import ResNet, ResBlock
from mcts_alpha import Node, AlphaMCTS

# Begin a new game
game = ChessEnv()
game.new_game()

args = {
	'C': 2,
	'num_searches': 1000,
	'num_iterations': 2,
	'num_self_play_iterations': 1,
	'num_epochs': 19,
	'batch_size': 64,
	'temperature': 1.25,
	'dirichlet_epsilon': 0.25,
	'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 18, 7, 256, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

alpha_zero = AlphaZero(game, args, model, optimizer)

alpha_zero.learn()
