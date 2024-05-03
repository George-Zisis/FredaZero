import numpy as np
import torch
import torch.multiprocessing as mp

from connect4 import ConnectFour
from gomoku import Gomoku
from mcts import VanillaMCTS, MCTS
from model import ResNet
from agent import AlphaZeroParallel
from evaluate import Evaluation


import warnings

warnings.filterwarnings("ignore", ".*Applied workaround*.",)


if __name__ == "__main__":
	mp.set_start_method("spawn")

	game = ConnectFour()

	args={
		'C': 2,
		'numSearches': 100,
		'spIterations': 20,
		'epochs': 4,
		'batchSize': 128,
		'temperature': 1.25,
		'dirichletEpsilon': 0.25,
		'dirichletAlpha': 0.3,
		'numWorkers': 8,
	}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = ResNet(game, device, filters=128, resBlocks=9)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

	iteration = 2
	if iteration > 1:
		model.load_state_dict(torch.load(f"model/model{repr(game)}{iteration-1}.pt"))
		optimizer.load_state_dict(torch.load(f"optim/optim{repr(game)}{iteration-1}.pt"))
	model.share_memory()

	mcts = MCTS(game, args, model)

	agent = AlphaZeroParallel(model, optimizer, game, args, mcts)

	agent.get_dataset(iteration)
	agent.train_dataset(iteration)

	evaluation = Evaluation(game, args, model, mcts)
	# evaluation.evaluateMCTS()
	# states = evaluation.get_states()
	# for state in states:
	# 	evaluation.evaluate_state(state)
	# evaluation.vsHuman()
	# evaluation.evaluateDataset(iteration)