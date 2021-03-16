import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, \
    QLabel, QFileDialog

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
    """Open the TrainerInterface window.
    """
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

    default_values = {p: trainer.DEFAULT_VALUES[p] for p in trainer.PARAMETERS}
   
    # Display parameters
    entries = {}
    for i, (param, value) in enumerate(default_values.items()):
        y = 20 + 50*i

        label = QLabel(param, window)
        label.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, y, 120, 40))

        entry = QLineEdit(str(value), window)
        entry.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 150, y, 220, 40))
        entry.setObjectName('param')
        entries[param] = entry

    # Add button to reset the model
    button = QPushButton('New Model', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 110, 170, 40))
    button.clicked.connect(lambda: new_model(trainer, entries))

    # Add button to load a model
    button = QPushButton('Load model', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 205, APP_PIXDIM[1] - 110, 170, 40))
    button.clicked.connect(lambda: load_model(trainer, entries))
    
    # Add button to launch the trainig to the interface
    button = QPushButton('Train', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0] + 25, APP_PIXDIM[1] - 60, 350, 40))
    button.clicked.connect(lambda: launch_training(trainer, entries))

    # Launch the PyQt programm
    window.show()
    qapp.exec_()


def launch_training(trainer, entries):
    """Launch the training from the trainer.
    - trainer: instance of Trainer to use for the training
    - entries: dictionary containing instance of QLineEdit and their param
    """
    # Convert params
    params = {}
    for param, widget in entries.items():
        value = widget.text()

        if param == 'name':
            params[param] = value

        elif param == 'hidden_layers':
            params[param] = eval(value)

        elif param in ['batch_size', 'nb_games', 'update_frequency']:
            params[param] = int(value)

        else:
            params[param] = float(value)

    trainer.set_parameters(**params)
    trainer.run()

    _update_params_display(trainer, entries)


def load_model(trainer, entries):
    """Load a model in the trainer.
    - trainer: instance of Trainer
    - entries: dictionary containing instance of QLineEdit and their param
    """
    uri = QFileDialog().getOpenFileName(caption="Select your model.zip",
                                        directory='./saves/',
                                        filter="*.zip")[0]
    if uri == "":
        # The user close the dialog without pick any file.
        return 

    trainer.load(uri)
    _update_params_display(trainer, entries)


def new_model(trainer, entries):
    """Reset the trainer to a new model. Reset entries as well.
    - trainer: instance of Trainer 
    - entries: dictionary containing instance of QLineEdit and their param
    """
    trainer.__init__(trainer.env)
    _update_params_display(trainer, entries)


def _update_params_display(trainer, entries):
    """Set the value of entries' paramaters to the trainer parameters values.
    - trainer: instance of Trainer 
    - entries: dictionary containing instance of QLineEdit and their param
    """
    parameters = trainer.get_parameters()
    for param, widget in entries.items():
        widget.setText(str(parameters[param]))
