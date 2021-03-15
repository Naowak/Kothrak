import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog

from kothrak.envs.KothrakEnv import KothrakEnv
from kothrak.envs.game.Utils import APP_PIXDIM
from dqn.Trainer import Trainer


STYLE = '''
QWidget#game_bg {
    background-color: rgb(180, 180, 180);
} 

QWidget#trainer_bg{
    background-color: rgb(100, 100, 100);
}

QLabel#message {
    background-color: rgba(255, 255, 255, 0);
    color: rgb(85, 85, 85);
    font: 30pt;
}

QLineEdit#param {
    background-color: rgb(180, 180, 180);
}

QPushButton {
    background-color: rgb(180, 180, 180);
}
'''


def run():    
    # Create the main window
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(STYLE)
    window = QWidget()
    window.resize(APP_PIXDIM[0] + 400, APP_PIXDIM[1])
    window.setWindowTitle('Kothrak training')
    window.setObjectName('trainer_bg')

    # Load the environnement
    env = KothrakEnv(qapp, window)
    trainer = Trainer(env)

    params_removed = ['nb_iter_prev', 'memory']
    default_parameters = {k: v for k, v in trainer.DEFAULT_PARAMETERS.items()
                             if k not in params_removed}
    
    # Display parameters
    entries = {}
    for i, (param, value) in enumerate(default_parameters.items()):
        y = 30 + 50*i

        label = QLabel(param, window)
        label.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, y, 120, 40))

        entry = QLineEdit(str(value), window)
        entry.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 150, y, 220, 40))
        entry.setObjectName('param')
        entries[param] = entry

    # Add button to reset the model
    button = QPushButton('New Model', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 120, 170, 40))
    button.clicked.connect(lambda: new_model(trainer, entries))

    # Add button to load a model
    button = QPushButton('Load model', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 205, APP_PIXDIM[1] - 120, 170, 40))
    button.clicked.connect(lambda: load_model(trainer, entries))
    
    # Add button to launch the trainig to the interface
    button = QPushButton('Train', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 70, 350, 40))
    button.clicked.connect(lambda: launch_training(trainer, entries))

    # Launch the PyQt programm
    window.show()
    qapp.exec_()


def launch_training(trainer, entries):
    # Convert params
    params = {}
    for param, widget in entries.items():
        value = widget.text()

        if param == 'name':
            params[param] = value

        # elif param == 'hidden_units':
        #     params[param] = eval(value)

        elif param in ['batch_size', 'nb_games']:
            params[param] = int(value)

        else:
            params[param] = float(value)

    trainer.set_parameters(**params)
    trainer.run()

    update_params_display(trainer, entries)


def load_model(trainer, entries):
    uri = QFileDialog().getOpenFileName(caption="Select your model.zip",
                                        filter="*.zip")[0]
    if uri == "":
        # The user close the dialog without pick any file.
        return 

    trainer.load(uri)
    update_params_display(trainer, entries)

def new_model(trainer, entries):
    trainer.__init__(trainer.env)
    update_params_display(trainer, entries)


def update_params_display(trainer, entries):
    parameters = trainer.get_parameters()
    for param, widget in entries.items():
        widget.setText(str(parameters[param]))
