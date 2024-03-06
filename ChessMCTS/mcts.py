import numpy as np
import math
from tqdm import tqdm


class Node:
	def __init__(self, game, args, state, parent=None, action_taken=None, action_idx=None):
		self.game = game
		self.args = args
		self.state = state
		self.parent = parent
		self.action_idx = action_idx  
		self.action_taken = action_taken

		self.children = []
		self.expandable_moves = game.get_valid_moves(state)
		self.expandable_moves_idx = np.ones(len(game.get_valid_moves(state)))

		self.visit_count = 0
		self.value_sum = 0

	def is_fully_expanded(self):
		return np.sum(self.expandable_moves_idx) == 0 and len(self.children) > 0

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
		action_idx = np.random.choice(np.where(self.expandable_moves_idx == 1)[0])
		action = self.expandable_moves[action_idx]
		self.expandable_moves_idx[action_idx] = 0

		self.game.board.push(action)
		child = Node(self.game, self.args, self.game.board.fen(), self, action, action_idx)
		self.children.append(child)
		return child

	def simulate(self):
		value, is_terminal = self.game.get_value_and_terminated(self.state)
		if is_terminal:
			return value
		rollout_state = self.state
		while True:
			valid_moves = self.game.get_valid_moves(rollout_state)
			action = np.random.choice(valid_moves)
			self.game.board.push(action)
			rollout_state = self.game.board.fen()
			value, is_terminal = self.game.get_value_and_terminated(rollout_state)
			if is_terminal:
				return value
			
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

		for search in tqdm(range(self.args['num_searches']), leave=True):
			node = root
			while node.is_fully_expanded():
				node = node.select()
				self.game.board.push(node.action_taken)

			value, is_terminal = self.game.get_value_and_terminated(node.state)

			if not is_terminal:
				node = node.expand()
				value = node.simulate()

			node.backpropagate(value)

			# Reset the game to initial state
			self.game.new_game()
			self.game.board.set_fen(state)

		action_probs = np.zeros(len(self.game.get_valid_moves(state)))
		for child in root.children:
			action_probs[child.action_idx] = child.visit_count
			action_probs = action_probs / np.sum(action_probs)
			return action_probs
