import numpy as np
import math

from tqdm import trange


class Node:
	def __init__(self, game, args, state, parent=None, action_idx=0, action_taken=None):
		self.game = game
		self.args = args
		self.state = state
		self.parent = parent
		self.action_idx = action_idx
		self.action_taken = action_taken

		self.children = []
		self.expandable_moves = np.ones(len(game.get_valid_actions(state)))

		self.visit_count = 0
		self.value_sum = 0

	def is_fully_expanded(self):
		return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

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
		q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
		return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

	def expand(self):
		valid_actions = self.game.get_valid_actions(self.state)
		action_idx = np.random.choice(np.where(self.expandable_moves == 1)[0])
		self.expandable_moves[action_idx] = 0
		action = valid_actions[action_idx]

		child_state = self.state
		child_state = self.game.get_next_state(child_state, action)

		child = Node(self.game, self.args, child_state, self, action_idx, action)
		self.children.append(child)

		return child

	def simulate(self):
		value, is_terminal = self.game.game_over(self.state)
		if is_terminal:
			return value

		rollout_state = self.state
		while True:
			value, is_terminal = self.game.game_over(rollout_state)
			if is_terminal:
				return value
			valid_moves = self.game.get_valid_actions(rollout_state)
			action = np.random.choice(valid_moves)
			rollout_state = self.game.get_next_state(rollout_state, action)

	def backpropagate(self, value):
		self.value_sum += value
		self.visit_count += 1

		if self.parent is not None:
			self.parent.backpropagate(value)


class MCTS:
	def __init__(self, game, args):
		self.game = game
		self.args = args

	def search(self, state):
		root = Node(self.game, self.args, state)

		for search in trange(self.args['num_searches']):
			node = root

			while node.is_fully_expanded():
				node = node.select()

			value, is_terminal = self.game.game_over(node.state)

			if not is_terminal:
				node = node.expand()
				value = node.simulate()

			node.backpropagate(value)

		action_probs = np.zeros(len(root.children))
		for child in root.children:
			action_probs[child.action_idx] = child.visit_count
		action_probs = action_probs / np.sum(action_probs)

		return action_probs
