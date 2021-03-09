import sys
import gym
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

from kothrak.envs.game.MyApp import MyApp, style, run
from kothrak.envs.game.Utils import APP_PIXDIM
from dqn.Trainer import Trainer

NB_GAMES = 100

def main():    
    # Create the main window
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.resize(1000, APP_PIXDIM[1])
    window.setWindowTitle('Kothrak training')

    # Load the environnement
    game = MyApp(window)
    env = gym.make('kothrak-v0')
    env.set_game(game)

    # Get the run name in the args
    run_name = ''
    if len(sys.argv) > 1:
        run_name = sys.argv[1]

    # Create the trainer
    trainer = Trainer(qapp, env, run_name=run_name, loading_file='saves/test-trainer.zip')

    # Add button to launch the trainig to the interface
    button = QPushButton('Play N Games', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0], 100, 100, 40))
    button.clicked.connect(lambda: trainer.run_n_games(NB_GAMES))

    # Launch the PyQt programm
    window.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    # values = ['ENV', 'GAME']
    RUN = 'ENV'

    if RUN == 'GAME':
        run()

    elif RUN == 'ENV':
        main()
