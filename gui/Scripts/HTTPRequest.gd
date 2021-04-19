extends HTTPRequest


func _ready():
	# warning-ignore:return_value_discarded	
	connect("request_completed", self, "_on_request_completed")


# Request agents infos to server
func request_agents_infos():
	# warning-ignore:return_value_discarded
	request('http://127.0.0.1:5000/agents_infos')

# Request new game to the server
func request_new_session(mode):
	# Change mode 
	Utils.MODE = mode
	
	# Retrieve params and update some Utils values
	var params = ''
	if mode == 'PvP':
		var nb_players = $"../Panel/Control_PvP/SpinBox_nbplayers".value
		var grid_ray = $"../Panel/Control_PvP/SpinBox_gridray".value
		params += 'nb_players=' + str(nb_players)
		params += '&grid_ray=' + str(grid_ray)
		Utils.NB_PERSON = nb_players
		Utils.NB_AGENTS = 0
		Utils.PLAYER_NAMES = []
		for i in range(nb_players):
			Utils.PLAYER_NAMES.append('Player_' + str(i))
		# warning-ignore:return_value_discarded	
		request("http://127.0.0.1:5000/new_game?" + params)
		
	elif mode == 'PvIA':
		var nb_person = $"../Panel/Control_PvIA/SpinBox_nbperson".value
		var grid_ray = $"../Panel/Control_PvIA/SpinBox_gridray".value
		var nb_agents = $"../Panel/Control_PvIA/SpinBox_nbIA".value
		if nb_person + nb_agents > 4:
			print('Error: you ask for more than 4 players.')
			return
		params += 'nb_players=' + str(nb_person + nb_agents)
		params += '&grid_ray=' + str(grid_ray)
		Utils.NB_PERSON = nb_person
		Utils.NB_AGENTS = nb_agents
		Utils.PLAYER_NAMES = []
		for i in range(nb_person):
			Utils.PLAYER_NAMES.append('Person_' + str(i))
		for i in range(nb_agents):
			Utils.PLAYER_NAMES.append('Agent_' + str(i))
		# warning-ignore:return_value_discarded	
		request("http://127.0.0.1:5000/new_game?" + params)
	
	elif mode == 'IAvIA':
		var nb_agents = $"../Panel/Control_IAvIA/SpinBox_nbagents".value
		var grid_ray = $"../Panel/Control_IAvIA/SpinBox_gridray".value
		var names = $'../Panel/Control_IAvIA/LineEditAgentNames'.text.replace(' ', '')
		if nb_agents > 4:
			print('Error: you ask for more than 4 players.')
			return
		if len(names.split(',')) != nb_agents:
			print(nb_agents, ' agents were ask but only ', len(names.split(',')), ' names are given.')
			return
		params += 'nb_agents=' + str(nb_agents)
		params += '&grid_ray=' + str(grid_ray)
		params += '&agent_names=' + names
		Utils.NB_PERSON = 0
		Utils.NB_AGENTS = nb_agents
		Utils.PLAYER_NAMES = []
		for n in names.split(','):
			Utils.PLAYER_NAMES.append(n.strip_edges())
		# warning-ignore:return_value_discarded
		request("http://127.0.0.1:5000/train?" + params)
		

# Request play to the server
func request_human_play(gid, play):
	var params = 'gid=' + str(gid)
	params += '&move=' + str(play['move'][0]) + ',' + str(play['move'][1])
	if play['build'] != null:
		params += '&build=' + str(play['build'][0]) + ',' + str(play['build'][1])
	else:
		params += '&build=' + str(play['build'])
	# warning-ignore:return_value_discarded
	request("http://127.0.0.1:5000/human_play?" + params)


# Request the server to make the next play
func request_agent_play(gid, agent_name):
	var params = 'gid=' + str(gid)
	params += '&agent_name=' + str(agent_name)
	# warning-ignore:return_value_discarded
	request('http://127.0.0.1:5000/agent_play?' + params)


# Request the server to get the history of last game
func request_watch_training(tid):
	var params = 'tid=' + str(tid)
	# warning-ignore:return_value_discarded
	request('http://127.0.0.1:5000/watch_training?' + params)


# Called when a request is completed : decode data and call _update from Playground
func _on_request_completed(_result, _response_code, _headers, body):
	var msg = body.get_string_from_utf8()
	var data = null
	if len(msg) > 0:
		data = JSON.parse(msg).result
	if data == null:
		data = {}
	data = _decode(data)
	get_parent()._update(data)


# Return new data where string keys with integer values are casted in integers
func _decode(data):
	var new_data = null
	if typeof(data) == TYPE_DICTIONARY:
		new_data = {}
		for key in data.keys():
			if key.is_valid_integer():
				new_data[int(key)] = _decode(data[key])
			else:
				new_data[key] = _decode(data[key])
	elif typeof(data) == TYPE_ARRAY:
		new_data = []
		for value in data:
			new_data += [_decode(value)]
	elif typeof(data) == TYPE_REAL and fposmod(data, 1) == 0:
		new_data = int(data)
	else:
		new_data = data
	return new_data
