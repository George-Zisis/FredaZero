from scipy.signal import convolve2d

import numpy as np

class Gomoku:
	def __init__(self, rows=6, columns=6, nPieces=4):
		self.rows = rows
		self.columns = columns

		self.nPieces = nPieces

		self.actionSpace = rows * columns

		self.winner = None

	def __repr__(self):
		return "Gomoku"

	def get_initial_state(self):
		return np.zeros((self.rows, self.columns))

	def get_next_state(self, state, action):
		player = self.get_current_player(state)
		row = action // self.rows
		column = action % self.columns
		state[row, column] = 1 if player == "WHITE" else -1
		return state 

	def get_current_player(self, state):
		posOnes = np.count_nonzero(state == 1)
		negOnes = np.count_nonzero(state == -1)
		return "BLACK" if posOnes > negOnes else "WHITE"

	def get_valid_moves(self, state):
		return (state.reshape(-1) == 0).astype(np.uint8)

	def get_encoded_state(self, state):
		return np.stack((state==1, state==0, state==-1)).astype(np.float32)

	def winning_move(self, state):
		# Create kernels for horizontal, vertical and diagonal win detection
		horizontal_kernel = np.ones((1, self.nPieces))
		vertical_kernel = np.transpose(horizontal_kernel)
		diag1_kernel = np.eye(self.nPieces, dtype=np.uint8)
		diag2_kernel = np.fliplr(diag1_kernel)

		detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

		# Use convolve2d function, nPieces indicates there were nPieces connected tiles in a row
		for kernel in detection_kernels:
			if (convolve2d(state, kernel, mode="valid") == self.nPieces).any():
					return True
		return False

	def check_game_over(self, state):
		# Set all of a player's tiles to 1, everything else 0
		white_board = (state==1).astype(np.float32)
		black_board = (state==-1).astype(np.float32)

		if np.sum(self.get_valid_moves(state)) == 0:
			self.winner = "DRAW"
			return 0, True

		if self.winning_move(white_board):
			self.winner = "WHITE"
			return 1, True

		if self.winning_move(black_board):
			self.winner = "BLACK"
			return -1, True

		self.winner = None
		return 0, False

	def change_perspective(self, state):
		return state * -1