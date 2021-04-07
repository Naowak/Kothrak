extends HTTPRequest


func _ready():
	# warning-ignore:return_value_discarded	
	connect("request_completed", self, "_on_request_completed")


# Request new game to the server
func request_new_game(mode):
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
		Utils.NB_IA = 0
		
	elif mode == 'PvIA':
		var nb_person = $"../Panel/Control_PvIA/SpinBox_nbperson".value
		var grid_ray = $"../Panel/Control_PvIA/SpinBox_gridray".value
		var nb_ia = $"../Panel/Control_PvIA/SpinBox_nbIA".value
		if nb_person + nb_ia > 4:
			print('Error: you ask for more than 4 players.')
			return
		params += 'nb_players=' + str(nb_person + nb_ia)
		params += '&grid_ray=' + str(grid_ray)
		Utils.NB_PERSON = nb_person
		Utils.NB_IA = nb_ia
	
	elif mode == 'IAvIA':
		pass
	
	print(params)
	# warning-ignore:return_value_discarded	
	request("http://127.0.0.1:5000/new_game?" + params)
	

# Request play to the server
func request_play(gid, play):
	var params = 'gid=' + str(gid)
	params += '&move=' + str(play['move'][0]) + ',' + str(play['move'][1])
	if play['build'] != null:
		params += '&build=' + str(play['build'][0]) + ',' + str(play['build'][1])
	else:
		params += '&build=' + str(play['build'])
	# warning-ignore:return_value_discarded
	request("http://127.0.0.1:5000/play?" + params)


# Request the server to make the next play
func request_watch(gid):
	var params = 'gid=' + str(gid)
	# warning-ignore:return_value_discarded
	request('http://127.0.0.1:5000/watch?' + params)


# Called when a request is completed : decode data and call _update from Playground
func _on_request_completed(_result, _response_code, _headers, body):
	var data = JSON.parse(body.get_string_from_utf8()).result
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
