from time import sleep
import torch

from dqn.Agent import Agent, save_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():

    def __init__(self, env, agent_names=None, nb_agents=2, nb_games=2000, 
            time_to_sleep=0, replay_size=20):
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
        self._init_agents(agent_names)


    def run(self):
        """Play nb_games withr all the agents.
        """
        # Run nb_games
        for n in range(self.nb_games):
            self._run_one_game()

        # End of the training
        for agent in self.agents:
            save_agent(agent)
    

    def last_replay(self):
        return self.replay.pop(0)


    def _run_one_game(self):
        """Play one game with all the agents and record it in self.replay.
        """
        done = False
        state = None
        rewards = [0 for _ in range(self.nb_agents)]
        next_state, infos = self.env.reset()
        next_state = torch.tensor(next_state, device=device).view(1, -1)
        
        # Make the same ai for the same player
        first_id = (self.env.game.next_player_id-1) % self.nb_agents
        order = list(range(first_id, self.nb_agents)) + list(range(first_id))
        agents = [self.agents[i] for i in order]

        agents_plays = [None for _ in range(self.nb_agents)]
        history = [infos]
        turn = -1

        while not done:

            # Get current agent and update state
            turn += 1
            current_id = turn % self.nb_agents
            current_agent = agents[current_id]
            state = next_state

            # Update current agent if all agents have played and state
            if turn >= self.nb_agents:
                current_agent.update(*agents_plays[current_id], next_state, 
                                            rewards[current_id], done)
                rewards[current_id] = 0

            # Move
            action = current_agent.play(state, train=True)
            agents_plays[current_id] = (state, action)

            next_state, play_reward, done, infos = self.env.step(action.item())
            next_state = torch.tensor(next_state, device=device).view(1, -1)
            history += [infos]

            # Update reward for all agent
            for k, v in play_reward.items():
                rewards[k] += v
          
            # Wait time_to_sleep second so the user can view the state
            sleep(self.time_to_sleep)
        

        # End of the game, update all agents 
        for i, agent in enumerate(agents):
            if turn >= i:
                agent.update(*agents_plays[i], next_state, rewards[i], done)

        # Add game to replay
        self._add_to_replay(history)
        
        # Wait time_to_sleep second so the user can view the state
        sleep(self.time_to_sleep)


    def _init_agents(self, agent_names):
        """Create agents and give them names.
        """
        if agent_names is None:
            self.agents = [Agent(self.env.num_observations, 
                self.env.num_actions) for _ in range(self.nb_agents)]

            for i, agent in enumerate(self.agents):
                name = agent.name + f'--{i}'
                agent.set_parameters(name=name)

        else:
            self.agents = [Agent(self.env.num_observations, 
                self.env.num_actions, name=agent_names[i]) 
                for i in range(self.nb_agents)]


    def _add_to_replay(self, history):
        """Add a new GameHistory to the replay_buffer, remove the first ones if there
        is no more room in the buffer.
        - init_state : list of values describing the initialisation of the game
        - first_agent : id of the first agent
        - actions : list of number representing all the actions during the game
        """
        if len(self.replay) >= self.replay_size:
            self.replay.pop(0)
        self.replay += [history]



def launch_nogui():
    from envs.KothrakEnv import KothrakEnv

    env = KothrakEnv()
    trainer = Trainer(env)
    trainer.run()
