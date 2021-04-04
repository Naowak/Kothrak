from flask import Flask, request  # render_template, jsonify

from envs.KothrakEnv import KothrakEnv


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
            data += [cast(request.args[name])]
        else:
            raise Exception(f'No arg {name} in request.')
    return data




app = Flask(__name__)
manager = Manager()


@app.route('/new_game', methods=['GET'])
def new_game():
    env = KothrakEnv()
    gid = manager.add(env)

    state = env.reset(state_vectorized=False)
    data = {'gid': gid, 'state': state}
    print(data)
    return data


@app.route('/play', methods=['GET'])
def play():

    gid, action = retrieve_args(id=int, action=int)

    state, rewards, done, _ = manager[gid].step(action, state_vectorized=False)

    return {'state': state, 'rewards': rewards, 'done': done}



app.run()

if __name__ == "__main__":
    app.run(debug=True)
