import numpy as np

import state_mod
import random
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from mcts_alpha import Node, AlphaMCTS


class AlphaZero:
	def __init__(self, game, args, model, optimizer):
		self.game = game
		self.args = args 
		self.model = model
		self.optimizer = optimizer
		self.mcts = AlphaMCTS(game, args, model)

	def self_play(self):
		memory = []
		self.game.new_game()
		state = self.game.get_fen_state()

		while True:
			value, is_terminal = self.game.get_value_and_terminated(state)

			if not is_terminal:
				action_probs = self.mcts.search(state)
				policy = self.game.correct_policy(action_probs)		
				memory.append((state, policy))

			if is_terminal:
				return_memory = []
				for hist_state, hist_action_probs in memory:
					hist_outcome = value
					hist_state = state_mod.get_state_board(hist_state, self.game.is_flip(hist_state))
					return_memory.append((hist_state, hist_action_probs, hist_outcome))
				return return_memory

			temperature = action_probs ** (1 / self.args['temperature'])
			temperature = temperature / np.sum(temperature)
			
			action = np.random.choice(list(self.game.board.legal_moves), p=temperature)

			self.game.board.push(action)
			state = self.game.get_fen_state()

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
			for self_play_iteration in trange(self.args['num_self_play_iterations']):
				print(f"Iteration: {iteration} / Self-Played Games: {self_play_iteration}")
				memory += self.self_play()

			self.model.train()
			for epoch in range(self.args['num_epochs']):
				print(f"In training, epoch: {epoch}")
				self.train(memory)

			torch.save(self.model.state_dict(), f"saved_model_{iteration}.pt")
			torch.save(self.optimizer.state_dict(), f"saved_optimizer_{iteration}.pt")
