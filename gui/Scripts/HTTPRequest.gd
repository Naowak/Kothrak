extends HTTPRequest


func _ready():
	# warning-ignore:return_value_discarded	
	connect("request_completed", self, "_on_request_completed")


# Request new game to the server
func new_game():
	# warning-ignore:return_value_discarded	
	request("http://127.0.0.1:5000/new_game")


# Called when a request is completed : decode data and call _update from Playground
func _on_request_completed(_result, _response_code, _headers, body):
	var data = _decode(JSON.parse(body.get_string_from_utf8()).result)
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
		for value in new_data:
			new_data += [_decode(value)]
	else:
		new_data = data
	return new_data
