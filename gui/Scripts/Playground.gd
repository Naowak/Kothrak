extends Spatial

func decode(data):
	# transform keys to int
	var new_data = null
	if typeof(data) == TYPE_DICTIONARY:
		new_data = {}
		for key in data.keys():
			if key.is_valid_integer():
				new_data[int(key)] = decode(data[key])
			else:
				new_data[key] = decode(data[key])
	elif typeof(data) == TYPE_ARRAY:
		new_data = []
		for value in new_data:
			new_data += [decode(value)]
	else:
		new_data = data
	return new_data
	
func _ready():
# warning-ignore:return_value_discarded
	$HTTPRequest.connect("request_completed", self, "_on_request_completed")
	$HTTPRequest.request("http://127.0.0.1:5000/new_game")


## API events ##
# warning-ignore:unused_argument
# warning-ignore:unused_argument
# warning-ignore:unused_argument
func _on_request_completed(result, response_code, headers, body):
	var data = decode(JSON.parse(body.get_string_from_utf8()).result)
	$Map.instance_map(data['state']['cells_stage'])
	$Map.instance_border()
	
## Cell clicked events ##
func _on_cell_clicked(cell):
	$Map.grew(cell)


