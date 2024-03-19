import numpy as np

import torch

from network import ResNet

from agent import AlphaZero

from chess_game import Game

from policy_mod import get_policy, get_dirichlet_policy

game = Game()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 18, 7, 256, device)

model.load_state_dict(torch.load("v5/saved_model_1.pt"))

model.eval()

state = game.new_game()

num_turns = 1

while True:
	reward, terminated = game.game_over(state)

	if terminated:
		print(f"Game ended in {num_turns} turns / Winner is: {game.winner}")
		break

	num_turns += 1

	policy, value = get_policy(game, model, state, device)

	policy = policy[policy!=0]

	valid_actions = game.get_valid_actions(state)

	action = valid_actions[np.argmax(policy)]

	state = game.get_next_state(state, action)
