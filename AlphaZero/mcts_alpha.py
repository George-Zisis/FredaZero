import numpy as np

import state_mod
import torch
import chess 
import math

from tqdm import trange


class Node:
	def __init__(self, game, args, state, prior=0, parent=None, action_taken=None, action_idx=None, visit_count=0):
		self.game = game
		self.args = args
		self.state = state
		self.prior = prior
		self.parent = parent
		self.action_idx = action_idx  
		self.action_taken = action_taken

		self.children = []

		self.visit_count = visit_count
		self.value_sum = 0

	def is_fully_expanded(self):
		return len(self.children) > 0

	def select(self):
		best_child = None 
		best_ucb = -np.inf

		for child in self.children:
			ucb = self.get_ucb(child)
			if ucb > best_ucb:
				best_child = child
				best_ucb = ucb

		return best_child

	def get_ucb(self, child):
		if child.visit_count == 0:
			q_value = 0
		else:
			q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
		return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

	def expand(self, policy):
		all_moves_str, all_moves_int = self.game.get_move_table()
		for idx, prob in enumerate(policy):
			if prob > 0:
				action = all_moves_str[idx]
				action_idx = self.game.get_valid_moves().index(action)
				self.game.board.push(chess.Move.from_uci(action))
				child = Node(self.game, self.args, self.game.board.fen(), prob, self, action, action_idx)
				self.children.append(child)
				self.game.new_game()
				self.game.board.set_fen(self.state)

	def backpropagate(self, value):
		self.value_sum += value
		self.visit_count += 1

		if self.parent is not None:
			self.parent.backpropagate(value)


class AlphaMCTS:
	def __init__(self, game, args, model):
		self.game = game
		self.args = args
		self.model = model

	def get_policy_and_value(self, node):
		board_state = state_mod.get_state_board(node.state, flip=self.game.is_flip(node.state))
		policy, value = self.model(torch.tensor(board_state, device=self.model.device).unsqueeze(0))
		# Process Policy
		policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
		all_moves_str, all_moves_int = self.game.get_move_table()
		policy = policy * all_moves_int
		policy = policy / np.sum(policy)
		# Process Value
		value = value.item()
		return policy, value

	def add_noise(self, node):
		board_state = state_mod.get_state_board(node.state, flip=self.game.is_flip(node.state))
		policy, value = self.model(torch.tensor(board_state, device=self.model.device).unsqueeze(0))
		policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
		policy = ((1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] *
			np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.get_all_action_size()))
		all_moves_str, all_moves_int = self.game.get_move_table()
		policy = policy * all_moves_int
		policy = policy / np.sum(policy)
		node.expand(policy)

	@torch.no_grad
	def search(self, state):
		root = Node(self.game, self.args, state, visit_count=1)
		self.add_noise(root)
		for search in trange(self.args['num_searches']):
			node = root
			while node.is_fully_expanded():
				node = node.select()
				self.game.board.push(chess.Move.from_uci(node.action_taken))

			value, is_terminal = self.game.get_value_and_terminated(node.state)

			if not is_terminal:
				policy, value = self.get_policy_and_value(node)
				node.expand(policy)

			node.backpropagate(value)

			# Reset the game to initial state
			self.game.new_game()
			self.game.board.set_fen(state)

		action_probs = np.zeros(len(root.children))
		for child in root.children:
			action_probs[child.action_idx] = child.visit_count
		action_probs = action_probs / np.sum(action_probs)
		return action_probs
