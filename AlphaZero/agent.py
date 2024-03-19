import numpy as np
import state_mod
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from mcts_alpha import AlphaMCTS
from policy_mod import get_policy, get_dirichlet_policy


class AlphaZero:
	def __init__(self, game, args, model, optimizer, device): 
		self.game = game 
		self.args = args 

		self.model = model
		self.optimizer = optimizer

		self.mcts = AlphaMCTS(game, args, model, device)

		self.device = device

	def selfplay(self):
		memory = []

		state = self.game.new_game()

		while True:
			reward, terminated = self.game.game_over(state)

			if terminated:
				return_memory = []
				for hist_state, hist_probs in memory:
					hist_reward = reward
					return_memory.append((hist_state, hist_probs, hist_reward))
				return return_memory

			action_probs = self.mcts.search(state)

			policy = self.game.valid_to_all_actions(state, action_probs)

			memory.append((state_mod.get_state_board(state, flip=False), policy))

			action_probs = action_probs ** (1 / self.args['temperature'])
			action_probs = action_probs / np.sum(action_probs)

			valid_actions = self.game.get_valid_actions(state)

			action = np.random.choice(valid_actions, p=action_probs)

			state = self.game.get_next_state(state, action)

	def train(self, memory):
		random.shuffle(memory)
		for batch_idx in range(0, len(memory), self.args['batch_size']):
			sample = memory[batch_idx: min(len(memory) - 1, batch_idx + self.args['batch_size'])]

			states, policy_targets, value_targets = zip(*sample)

			states = np.array(states, dtype=np.float32) 
			policy_targets = np.array(policy_targets, dtype=np.float32) 
			value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

			states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
			policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
			value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

			out_policy, out_value = self.model(states)

			policy_loss = F.cross_entropy(out_policy, policy_targets)
			value_loss = F.mse_loss(out_value, value_targets)
			loss = policy_loss + value_loss

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def learn(self):
		for iteration in range(self.args['num_iterations']):
			memory = []

			self.model.eval()
			for sp_iteration in trange(self.args['num_self_play_iterations']):
				memory += self.selfplay()

			self.model.train()
			for epoch in trange(self.args['num_epochs']):
				self.train(memory)

			torch.save(self.model.state_dict(), f"v5/saved_model_{iteration}.pt")
			torch.save(self.optimizer.state_dict(), f"v5/saved_optimizer_{iteration}.pt")
