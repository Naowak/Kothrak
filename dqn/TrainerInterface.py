import sys
import gym
from datetime import datetime
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog

from kothrak.envs.game.MyApp import MyApp
from kothrak.envs.game.Utils import APP_PIXDIM
from dqn.Trainer import Trainer


STYLE = '''
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

QPushButton {
    background-color: rgb(180, 180, 180);
}

QLabel#cell* {
    background-color: rgba(255, 255, 255, 0);
}

QLabel#player* {
    background-color: rgba(255, 255, 255, 0);
}
'''

NB_GAMES = 300

DEFAULT_PARAMS = {'run_name': datetime.now().strftime("%m%d%y-%H%M"),
            'epsilon': 0.99, 'decay': 0.9998, 
            'min_epsilon': 0, 'lr': 1e-3, 'gamma': 0.99, 'batch_size': 32,
            'min_experiences': 100, 'max_experiences': 2000, 
            'hidden_units': [120, 120, 120, 120]}

def run():    
    # Create the main window
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(STYLE)
    window = QWidget()
    window.resize(APP_PIXDIM[0] + 400, APP_PIXDIM[1])
    window.setWindowTitle('Kothrak training')
    window.setObjectName('trainer_bg')

    # Load the environnement
    game = MyApp(window)
    env = gym.make('kothrak-v0')
    env.set_game(game)
    
    # Display parameters
    entries = {}
    for i, (param, value) in enumerate(DEFAULT_PARAMS.items()):
        y = 30 + 50*i

        label = QLabel(param, window)
        label.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, y, 120, 40))

        entry = QLineEdit(str(value), window)
        entry.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 150, y, 220, 40))
        entry.setObjectName('param')
        entries[param] = entry

    trainer = Trainer(qapp, env)

    # Add button to load a model
    button = QPushButton('Load model', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 100, 170, 40))
    button.clicked.connect(lambda: load_model(trainer, entries))
    
    # Add button to launch the trainig to the interface
    button = QPushButton('Play N Games', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 205, APP_PIXDIM[1] - 100, 170, 40))
    button.clicked.connect(lambda: launch_training(trainer, entries))

    # Launch the PyQt programm
    window.show()
    sys.exit(qapp.exec_())


def launch_training(trainer, entries):
    params = {}
    for param, widget in entries.items():
        value = widget.text()

        if param == 'run_name':
            params[param] = value

        elif param == 'hidden_units':
            params[param] = eval(value)

        elif param in ['batch_size', 'min_experiences', 'max_experiences']:
            params[param] = int(value)

        else:
            params[param] = float(value)

    if trainer.run_name == '':
        # Create the trainer
        trainer.new_session(**params)
    else:
        # Trainer has been load, change parameters if user change something
        trainer.set_params(**params)

    trainer.run_n_games(NB_GAMES)


def load_model(trainer, entries):
    uri = QFileDialog().getOpenFileName(caption="Select your model.zip",
                                        filter="*.zip")[0]
    if uri == "":
        # The user close the dialog without pick any file.
        return 

    trainer.load_session(uri)

    # update entries
    parameters = trainer.get_params()
    for param, widget in entries.items():
        widget.setText(str(parameters[param]))
