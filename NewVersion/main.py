import warnings
warnings.filterwarnings("ignore", ".*Applied workaround*.",)

import torch
import multiprocessing

from core.mcts import MonteCarloTreeSearch
from core.model import ResNet
from core.agent import AlphaZero
from core.evaluator import Evaluate

from config import get_game_and_args

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    game, args = get_game_and_args()

    print("===> Building Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, device, input_channels=args.input_channels, filters=args.filters, res_blocks=args.res_blocks)
    print(f"...Device: {device}")
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.iteration > 1:
        model.load_state_dict(torch.load(f"./model/model{repr(game)}{args.iteration-1}.pt"))
        optimizer.load_state_dict(torch.load(f"./optim/optim{repr(game)}{args.iteration-1}.pt"))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    mcts = MonteCarloTreeSearch(game, args, model)
    
    agent = AlphaZero(model, optimizer, scheduler, game, args, mcts)
    if args.get_dataset:
        agent.create_dataset(datasetID=args.iteration, parallel=args.parallelize)
    if args.train_model:
        agent.train_dataset(datasetID=args.iteration)
    
    evaluate = Evaluate(game, args, model, mcts)
    if args.play_vs_bot:
        evaluate.vs_human()
    if args.bot_vs_previous_bot:
        evaluate.vs_bot(1, 2)
