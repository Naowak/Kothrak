from flask import Flask
import torch

from envs.KothrakEnv import KothrakEnv, transform_actions_into_number
from dqn.Player import Player
from api.Utils import Manager, retrieve_args, cors, load_agents


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = Flask(__name__)
manager = Manager()
player_ia = None  # Should load all DQNs in saves/
agents = load_agents()


@api.route('/new_game', methods=['GET'])
def new_game():
    global player_ia, manager
    nb_players, grid_ray = retrieve_args(nb_players=int, grid_ray=int)
    env = KothrakEnv(nb_players, grid_ray)
    gid = manager.add(env)
    if player_ia is None:
        player_ia = Player(env.num_observations, env.num_actions)

    _, infos = env.reset()
    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/agent_play', methods=['GET'])
def agent_play():

    [gid] = retrieve_args(gid=int)
    state, _ = manager[gid]._get_observation()
    state = torch.tensor(state, device=device).view(1, -1)

    if player_ia.last_state is not None:
        player_ia.update(state, 0, 0)  # /!\ NEEDS TO BE MODIFIED

    action = player_ia.play(state)
    _, _, done, infos = manager[gid].step(action)

    data = {'gid': gid, **infos}
    print(data)
    return cors(data)



@api.route('/human_play', methods=['GET'])
def human_play():
    gid, move, build = retrieve_args(gid=int, move='cell', build='cell')
    action = transform_actions_into_number(move, build)
    _, _, done, infos = manager[gid].step(action)
    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/train', methods=['GET'])
def train():
    pass



if __name__ == "__main__":
    api.run(debug=True)
