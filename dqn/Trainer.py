from collections import namedtuple
from time import sleep
import random
import torch

from dqn.Agent import Agent, save_agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GameHistory = namedtuple('GameHistory',
                        ('init_state', 'first_agent', 'actions'))


class Trainer():

    def __init__(self, env, nb_agents=2, nb_games=2000, time_to_sleep=0,
            replay_size=20):
        """Initialize the Trainer.
        - env : KothrakEnv instance
        """ 
        self.env = env
        self.nb_agents = nb_agents
        self.nb_games = nb_games
        self.time_to_sleep = time_to_sleep
        self.replay_size = 20
        self.replay = []

        # agents
        self.agents = None
        self._init_agents()


    def run(self):
        """Play nb_games withr all the agents.
        """
        # Run nb_games
        for n in range(self.nb_games):
            self._run_one_game()

        # End of the training
        for agent in self.agents:
            save_agent(agent)
    

    def replay(self):
        return self.replay_size.pop(0)


    def _run_one_game(self):
        """Play one game with all the agents and record it in self.replay.
        """
        init_state, _ = self.env.reset()
        init_state = torch.tensor(init_state, device=device).view(1, -1)
        first_agent_id = random.randrange(self.nb_agents)
        actions = []

        state = None
        action = None
        next_state = init_state
        rewards = [0 for _ in range(self.nb_agents)]
        done = False

        current_id = first_agent_id - 1
        agents_last_play = [None for _ in range(self.nb_agents)]

        while not done:

            # Get current agent and update state
            current_id = (current_id+1) % self.nb_agents
            current_agent = self.agents[current_id]
            state = next_state

            # Update current agent if all agents have played and state
            if None not in agents_last_play:
                last_state, last_action = agents_last_play[current_id]
                current_agent.update(last_state, last_action, next_state, 
                                                rewards[current_id], done)
                rewards[current_id] = 0

            # Move
            action = current_agent.play(state)
            actions += [action]
            agents_last_play[current_id] = (state, action)

            next_state, agents_reward, done, _ = self.env.step(action.item())
            next_state = torch.tensor(next_state, device=device).view(1, -1)

            # Update reward for all agent
            for k, v in agents_reward.items():
                rewards[k] += v
          
            # Wait time_to_sleep second so the user can view the state
            sleep(self.time_to_sleep)
        

        # End of the game, update all agents 
        for i, agent in enumerate(self.agents):
            last_play = agents_last_play[i]
            if last_play is not None:
                agent.update(*last_play, next_state, rewards[i], done)

        # Add game to replay
        self._add_to_replay(init_state, first_agent_id, actions)
        
        # Wait time_to_sleep second so the user can view the state
        sleep(self.time_to_sleep)


    def _init_agents(self):
        """Create agents and give them names.
        """
        self.agents = [Agent(self.env.num_observations, self.env.num_actions) 
            for _ in range(self.nb_agents)]

        for i, agent in enumerate(self.agents):
            name = agent.name + f'--{i}'
            agent.set_parameters(name=name)


    def _add_to_replay(self, init_state, first_agent, actions):
        """Add a new GameHistory to the replay_buffer, remove the first ones if there
        is no more room in the buffer.
        - init_state : list of values describing the initialisation of the game
        - first_agent : id of the first agent
        - actions : list of number representing all the actions during the game
        """
        if len(self.replay) >= self.replay_size:
            self.replay.pop(0)
        self.replay += [GameHistory(init_state, first_agent, actions)]


            


def launch_nogui():
    from envs.KothrakEnv import KothrakEnv

    env = KothrakEnv()
    trainer = Trainer(env)
    trainer.run()
