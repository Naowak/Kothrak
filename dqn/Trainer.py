from time import sleep


import torch







class Trainer():

    def __init__(self, env, nb_games=1000, time_to_sleep=0):
        """Initialize the Trainer.
        - env : KothrakEnv instance
        """        
        self.env = env
        self.nb_games = None
        self.time_to_sleep = None


    def run(self):
        """Play nb_games.
        """

        # Run nb_games
        for n in range(self.nb_games):

            reward, loss = self._run_one_game()

        # End of the training
        self.nb_iter_prev += self.nb_games
        self.save()


    def _run_one_game(self):
        """Play one game and optimize model.
        """
        done = False
        state = torch.tensor(self.env.reset(), device=device).view(1, -1)

        while not done:

            action = self.player.play(state)
            next_state, reward, done, _ = self.env.step(action.item())
            self.player.update(next_state, reward, done) # /!\ SELF.PLAYER DOES NOT EXIST
            
            # Prepare next state
            state = next_state

            # Wait time_to_sleep second so the user can view the state
            sleep(self.time_to_sleep)
            

def launch_test():
    """Create an instance of trainer and launch the training to test the class
    """
    import sys
    from kothrak.KothrakEnv import KothrakEnv
    from kothrak.game.MyApp import style
    from PyQt5.QtWidgets import QApplication, QWidget

    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.setWindowTitle('Kothrak training')

    env = KothrakEnv(qapp, window)
    window.show()

    trainer = Trainer(env)
    # trainer.load('saves/031421-1523.zip')
    trainer.run()

    qapp.exec_()
