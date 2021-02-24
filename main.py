from envs.game import MyApp
from envs.KothrakEnv import KothrakEnv
import sys
import asyncio


def simulate():
    
    # things to put here ...
    # for episode in range()

    state = ''














if __name__ == '__main__':

    # values = ['ENV', 'GAME']
    RUN = 'GAME'

    if RUN == 'GAME':
        MyApp.run()

    elif RUN == 'ENV':
        kkenv = KothrakEnv()
        # kkenv.reset()