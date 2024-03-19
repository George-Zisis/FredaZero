import numpy as np
import torch

from network import ResNet
from chess_game import Game 
from agent import AlphaZero

game = Game()

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 2,
    'num_self_play_iterations': 1,
    'num_epochs': 4,
    'batch_size': 256,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 18, 7, 256, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

agent = AlphaZero(game, args, model, optimizer, device)

agent.learn()
