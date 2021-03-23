from kothrak.game.MyApp import run as run_game
from dqn.TrainerInterface import run as run_training

if __name__ == '__main__':

    RUN = 'TRAIN'

    if RUN == 'GAME':
        run_game()

    elif RUN == 'TRAIN':
        run_training()
