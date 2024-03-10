import numpy as np

import action_mod
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
	'num_iterations': 1,
	'num_self_play_iterations': 2,
	'num_epochs': 19,
	'batch_size': 64,
	'temperature': 1.25,
	'dirichlet_epsilon': 0.25,
	'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 18, 7, 256, device)

model.load_state_dict(torch.load("saved_model_1.pt"))
model.eval()

mcts = AlphaMCTS(game, args, model)

game.new_game()
state = game.get_fen_state()

while True:
	print(game.board)

	game.eval_stockfish(state)

	board_state = state_mod.get_state_board(state, flip=game.is_flip(state))
	policy, state_value = model(torch.tensor(board_state, device=model.device).unsqueeze(0))

	policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
	all_moves_str, all_moves_int = game.get_move_table()
	policy = policy + 0.00001
	policy = policy * all_moves_int
	policy = np.where(policy > 0)[0]
	policy = policy / np.sum(policy)

	state_value = state_value.item()

	print(f"State value according to model:{state_value}")

	value, is_terminal = game.get_value_and_terminated(state)

	if is_terminal:
		print(f"Game is over! Winner is {game.winner}, result:{game.result}, game lasted for {game.board.fullmove_number}")
		print(f"Number of halfmoves is {game.board.halfmove_clock}")
		print(f"Reward:{value}")
		break

	action = np.random.choice(list(game.board.legal_moves), p=policy)

	print(f"Best action is {action}")

	game.board.push(action)

	state = game.get_fen_state()
