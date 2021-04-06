from flask import Flask, request, jsonify

from envs.KothrakEnv import KothrakEnv, transform_actions_into_number


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


@app.route('/new_game', methods=['GET'])
def new_game():
    mode, nb_players, grid_ray = retrieve_args(mode=str, 
                                        nb_players=int, grid_ray=int)
    env = KothrakEnv(nb_players, grid_ray)
    gid = manager.add(env)

    _, infos = env.reset()
    data = {'gid': gid, 'mode': mode, **infos}
    return cors(data)


@app.route('/watch', methods=['GET'])
def watch():
    pass


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
