import sys
import gym
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel

from kothrak.envs.game.MyApp import MyApp
from kothrak.envs.game.Utils import APP_PIXDIM
from dqn.Trainer import Trainer

NB_GAMES = 1000

style = '''
QWidget#game_bg {
    background-color: rgb(180, 180, 180);
} 

QWidget#trainer_bg{
    background-color: rgb(100, 100, 100)
}

QLabel#message {
    background-color: rgba(255, 255, 255, 0);
    color:rgb(85, 85, 85);
    font:30pt;
}

QLineEdit#param {
    background-color: rgb(180, 180, 180);
}

QPushButton#play {
    background-color: rgb(180, 180, 180);
}

QLabel#cell* {
    background-color: rgba(255, 255, 255, 0);
}

QLabel#player* {
    background-color: rgba(255, 255, 255, 0);
}


'''

def launch_training(trainer, entries):
    params = {}
    for k, v in entries.items():
        value = v.text()

        if k == 'run_name':
            params[k] = value

        elif k == 'hidden_units':
            params[k] = eval(value)

        elif k in ['batch_size', 'min_experiences', 'max_experiences']:
            params[k] = int(value)

        else:
            params[k] = float(value)

    trainer.modify_params(**params)
    trainer.run_n_games(NB_GAMES)



def run():    
    # Create the main window
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.resize(APP_PIXDIM[0] + 400, APP_PIXDIM[1])
    window.setWindowTitle('Kothrak training')
    window.setObjectName('trainer_bg')

    # Load the environnement
    game = MyApp(window)
    env = gym.make('kothrak-v0')
    env.set_game(game)

    # Get the run name in the args
    run_name = ''
    if len(sys.argv) > 1:
        run_name = sys.argv[1]

    # Create the trainer
    trainer = Trainer(qapp, env, run_name=run_name, loading_file='')
    
    # Display parameters
    params = trainer.get_params()
    entries = {}
    for i, (k, v) in enumerate(params.items()):
        y = 30 + 50*i

        label = QLabel(k, window)
        label.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, y, 120, 40))

        entry = QLineEdit(str(v), window)
        entry.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 150, y, 220, 40))
        entry.setObjectName('param')
        entries[k] = entry
    
    # Add button to launch the trainig to the interface
    button = QPushButton('Play N Games', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 100, 350, 40))
    button.setObjectName('play')
    button.clicked.connect(lambda: launch_training(trainer, entries))

    # Launch the PyQt programm
    window.show()
    sys.exit(qapp.exec_())
