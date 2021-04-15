# from dqn.TrainerInterface import run as run_training
# from dqn.Trainer import launch_nogui

from api.API import api


if __name__ == '__main__':

    RUN = 'API'

    # if RUN == 'GAME':
    #     run_game()

    # elif RUN == 'TRAIN':
    #     run_training()

    # elif RUN == 'TEST_TRAINER':
    #     launch_nogui()

    if RUN == 'API':
        api.run()
