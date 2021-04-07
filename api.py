from flask import Flask, request, jsonify
import torch

from envs.KothrakEnv import KothrakEnv, transform_actions_into_number
from dqn.Player import Player

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Manager():

    def __init__(self, nb_slots=100):
        self.items = {}
        self.ids = list(range(nb_slots))

    def add(self, item):
        id_ = self.ids.pop(0)
        self.items[id_] = item
        return id_

    def remove(self, id_):
        del(self.items[id_])
        self.ids += [id_]

    def __getitem__(self, id_):
        return self.items[id_]


def retrieve_args(**args):
    """ args format : name : type (id : int)
    """
    data = []
    for name, cast in args.items():
        if name in request.args:
            if cast == 'cell':
                if request.args[name] == 'Null':
                    data += [None]
                else:
                    data += [[int(x) for x in request.args[name].split(',')]]
            else:                
                data += [cast(request.args[name])]
        else:
            raise Exception(f'No arg {name} in request.')
    return data


def cors(data):
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response




app = Flask(__name__)
manager = Manager()
player_ia = None


@app.route('/new_game', methods=['GET'])
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


@app.route('/watch', methods=['GET'])
def watch():

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



@app.route('/play', methods=['GET'])
def play():
    gid, move, build = retrieve_args(gid=int, move='cell', build='cell')
    action = transform_actions_into_number(move, build)
    _, _, done, infos = manager[gid].step(action)
    data = {'gid': gid, **infos}
    return cors(data)



app.run()

if __name__ == "__main__":
    app.run(debug=True)
