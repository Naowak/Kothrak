from kothrak.envs.game.MyApp import run as run_game
from dqn.TrainerInterface import run as run_training

if __name__ == '__main__':
    # values = ['TRAIN', 'GAME']
    RUN = 'TRAIN'

    if RUN == 'GAME':
        run_game()

    elif RUN == 'TRAIN':
        run_training()
