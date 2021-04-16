from flask import request, jsonify
import os

from dqn.Agent import load_agent

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


class Training():
    def __init__(self, trainer=None, thread=None):
        self.trainer = trainer
        self.thread = thread


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


def load_all_agents(existing_agents={}, saves_directory='saves/'):
    # Verify path
    if not os.path.exists(saves_directory):
        print(f'No directory {saves_directory}, no saves available.')
        return {}

    if saves_directory[-1] != '/':
        saves_directory += '/'

    # Load all agents
    agents = {}
    for filename in os.listdir(saves_directory):
        if filename[:-4] not in existing_agents.keys():
            path = saves_directory + filename
            agent = load_agent(path)
            agents[agent.name] = agent

    return agents
