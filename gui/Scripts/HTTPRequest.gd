extends HTTPRequest


func _ready():
	# warning-ignore:return_value_discarded	
	connect("request_completed", self, "_on_request_completed")


# Request new game to the server
func new_game(mode):
	# warning-ignore:return_value_discarded
	var params = 'mode=' + str(mode) 
	request("http://127.0.0.1:5000/new_game?" + params)
	

# Request play to the server
func play(gid, play):
	var params = 'gid=' + str(gid)
	params += '&move=' + str(play['move'][0]) + ',' + str(play['move'][1])
	if play['build'] != null:
		params += '&build=' + str(play['build'][0]) + ',' + str(play['build'][1])
	else:
		params += '&build=' + str(play['build'])
	# warning-ignore:return_value_discarded
	request("http://127.0.0.1:5000/play?" + params)


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
	else:
		new_data = data
	return new_data
