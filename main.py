from envs.game import MyApp
from envs.KothrakEnv import KothrakEnv
import sys
import asyncio

# values = ['ENV', 'GAME']
RUN = 'ENV'

if RUN == 'GAME':
    MyApp.run()

elif RUN == 'ENV':
    kkenv = KothrakEnv()
    kkenv.reset()