from flask import Flask
import torch

from envs.KothrakEnv import KothrakEnv, transform_actions_into_number
from api.Utils import Manager, retrieve_args, cors, load_all_agents
from dqn.Trainer import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = Flask(__name__)
games = Manager()
trainers = Manager()
agents = load_all_agents()


@api.route('/agents_infos', methods=['GET'])
def agents_infos():
    agents = load_all_agents()
    data = {'names': list(agents.keys())}
    return cors(data)


@api.route('/new_game', methods=['GET'])
def new_game():
    nb_players, grid_ray = retrieve_args(nb_players=int, grid_ray=int)
    env = KothrakEnv(nb_players, grid_ray)
    gid = games.add(env)

    _, infos = env.reset()
    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/agent_play', methods=['GET'])
def agent_play():
    gid, agent_name = retrieve_args(gid=int, agent_name=str)
    state, _ = games[gid]._get_observation()
    state = torch.tensor(state, device=device).view(1, -1)

    action = agents[agent_name].play(state)
    _, _, done, infos = games[gid].step(action)

    data = {'gid': gid, **infos}
    print(data)
    return cors(data)


@api.route('/human_play', methods=['GET'])
def human_play():
    gid, move, build = retrieve_args(gid=int, move='cell', build='cell')
    action = transform_actions_into_number(move, build)
    _, _, done, infos = games[gid].step(action)
    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/train', methods=['GET'])
def train():  # Il faudra la mettre dans un truc qui permet de parallèliser
    env = KothrakEnv(2, 2)
    trainer = Trainer(env)
    tid = trainers.add(trainer)

    trainers[tid].run()
    data = {'tid': tid}
    return cors(data)


@api.route('/watch_training', methods=['GET'])
def watch_training():
    tid = retrieve_args(tid=int)
    gamehist = trainers[tid].replay()
    data = {'tid': tid, 'watch': gamehist._asdict()}
    return cors(data)



if __name__ == "__main__":
    api.run(debug=True)
