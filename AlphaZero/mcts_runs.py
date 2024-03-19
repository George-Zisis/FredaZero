import numpy as np
import time
import torch

from network import ResNet
from chess_game import Game
from mcts_alpha import AlphaMCTS

game = Game()

args = {
	'C': 2,
	'num_searches': 10000,
	'dirichlet_epsilon': 0.25,
	'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 18, 7, 256, device)

mcts = AlphaMCTS(game, args, model, device)

def check_random_fens():
	fen_26 = "2R2n2/Nqrp4/P2PP2R/pP2b3/2Bp3p/1pp1P1k1/2PP1N1p/2K3B1 w - - 0 1"
	fen_22 = "2R5/P1r3p1/2bq1PQ1/7K/P1p4b/BPk1pr1P/1p1N2P1/3B4 w - - 0 1"
	fen_16 = "8/2P2p2/2P1P3/1k3P2/1p4B1/2q3Q1/Kp2r1P1/R3N3 w - - 0 1"
	fen_12 = "2N5/2KBp1P1/7n/4B3/3N4/1R4r1/6kP/8 w - - 0 1"
	fen_8 = "3k4/8/6R1/5Pr1/4K3/8/P3P3/N7 w - - 0 1"
	fen_4 = "7r/8/7P/1K1k4/8/8/8/8 w - - 0 1"

	fens = [fen_26, fen_22, fen_16, fen_12, fen_8, fen_4]

	for state in fens:
		valid_actions = game.get_valid_actions(state)

		stockfish_index = valid_actions.index(game.stockfish_action(state))

		probs = mcts.search(state)

		best_idx = np.argmax(probs)

		action = valid_actions[best_idx]

		print(f"The best action is in the {stockfish_index} index while mcts gives {best_idx} index")
		print(f"The best action prob is: {probs[stockfish_index]} while the mcts action prob is {probs[best_idx]}")

		print(f"The probs we get from mcts are: \n {probs}")

def selfplay():
	state = game.new_game()

	print(f"Player is {game.player(state)} and board score is {game.stockfish_score(state)}")

	num_turns = 1

	while True:
		reward, terminated = game.game_over(state)

		if terminated:
			print(f"Game ended in {num_turns} turns, winner is {game.winner}")
			break

		num_turns += 1

		print(f"Player is {game.player(state)} and board score is {game.stockfish_score(state)}")

		valid_actions = game.get_valid_actions(state)

		action_probs = mcts.search(state)

		action = np.random.choice(valid_actions, p=action_probs)
		
		state = game.get_next_state(state, action)

# check_random_fens()
selfplay()
