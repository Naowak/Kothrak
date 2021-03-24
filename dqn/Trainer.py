from time import sleep
import torch

from dqn.Player import Player


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():

    def __init__(self, env, nb_players=2, nb_games=50000, time_to_sleep=0):
        """Initialize the Trainer.
        - env : KothrakEnv instance
        """ 
        self.env = env
        self.nb_players = nb_players
        self.nb_games = nb_games
        self.time_to_sleep = time_to_sleep

        # Players
        self.players = None
        self._init_players()


    def run(self):
        """Play nb_games withr all the players.
        """
        # Run nb_games
        for n in range(self.nb_games):
            self._run_one_game()

        # End of the training
        for player in self.players:
            player.save()


    def _run_one_game(self):
        """Play one game with all the players.
        """
        done = False
        rewards = [0 for _ in range(self.nb_players)]
        state = None
        next_state = torch.tensor(self.env.reset(), device=device).view(1, -1)
        turn = -1

        while not done:

            # Get current player and update state
            turn += 1
            current_pid = turn % self.nb_players
            current_player = self.players[current_pid]
            state = next_state

            # Update current player if all players have played and state
            if turn >= self.nb_players:
                current_player.update(next_state, rewards[current_pid], done)
                rewards[current_pid] = 0

            # Move
            action = current_player.play(state)
            state, players_reward, done, _ = self.env.step(action.item())
            state = torch.tensor(state, device=device).view(1, -1)

            # Update reward for all player
            for k, v in players_reward.items():
                rewards[k] += v

            # If game over, update all players and quit the loop
            if done:
                for i, player in enumerate(self.players):
                    player.update(state, rewards[i], done)
                sleep(self.time_to_sleep)
                break
            # If game is not over, update only current player
            else: 
                current_player.update(state, rewards[current_pid], done)
                rewards[current_pid] = 0
            
            # Build (no rewards or done possible while building)
            action = current_player.play(state)
            next_state, _, _, _ = self.env.step(action.item())
            next_state = torch.tensor(state, device=device).view(1, -1)
            
            # Wait time_to_sleep second so the user can view the state
            sleep(self.time_to_sleep)


    def _init_players(self):
        """Create players and give them names.
        """
        self.players = [Player(self.env.num_observations, self.env.num_actions) 
            for _ in range(self.nb_players)]

        for i, player in enumerate(self.players):
            name = player.name + f'--{i}'
            player.set_parameters(name=name)
            

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

    env = KothrakEnv(qapp, window, state_mode='absolute')
    window.show()

    trainer = Trainer(env)
    trainer.run()

    qapp.exec_()
