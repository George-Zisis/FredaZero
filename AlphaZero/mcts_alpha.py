import numpy as np

import state_mod
import torch
import math

from policy_mod import get_policy, get_dirichlet_policy

from tqdm import trange


class Node:
	def __init__(self, game, args, state, prior=0, parent=None, action_taken=None, action_idx=0, visit_count=0):
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
		policy = policy[policy!=0]

		valid_actions = self.game.get_valid_actions(self.state)

		for action_idx, prob in enumerate(policy):
			if prob > 0:
				action = valid_actions[action_idx]
				state = self.game.get_next_state(self.state, action)
				child = Node(self.game, self.args, state, prob, self, action, action_idx)
				self.children.append(child)

	def backpropagate(self, value):
		self.value_sum += value
		self.visit_count += 1

		if self.parent is not None:
			self.parent.backpropagate(value)


class AlphaMCTS:
	def __init__(self, game, args, model, device):
		self.game = game
		self.args = args
		self.model = model
		self.device = device

	@torch.no_grad
	def search(self, state):
		root = Node(self.game, self.args, state, visit_count=1)

		policy, _ = get_dirichlet_policy(self.game, self.model, state, self.device, 
			dirichlet_epsilon=self.args['dirichlet_epsilon'], dirichlet_alpha=self.args['dirichlet_alpha'])

		root.expand(policy)

		for search in trange(self.args['num_searches']):
			node = root

			while node.is_fully_expanded():
				node = node.select()

			value, is_terminal = self.game.game_over(node.state)

			if not is_terminal:
				policy, value = get_policy(self.game, self.model, node.state, self.device)
				node.expand(policy)

			node.backpropagate(value)

		action_probs = np.zeros(len(root.children))

		for child in root.children:
			action_probs[child.action_idx] = child.visit_count

		action_probs = action_probs / np.sum(action_probs)

		return action_probs
