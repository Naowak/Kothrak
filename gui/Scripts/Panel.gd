extends Panel

# Called when the node enters the scene tree for the first time.
func _ready():
	# warning-ignore:return_value_discarded
	$Button_PvP.connect('pressed', self, '_on_control_change', ['PvP'])
	# warning-ignore:return_value_discarded
	$Button_PvIA.connect('pressed', self, '_on_control_change', ['PvIA'])
	# warning-ignore:return_value_discarded
	$Button_IAvIA.connect('pressed', self, '_on_control_change', ['IAvIA'])


func _on_control_change(mode):
	var buttons = {'PvP': $Button_PvP, 'PvIA': $Button_PvIA, 
		'IAvIA': $Button_IAvIA}
	var nodes_control = {'PvP': $Control_PvP, 'PvIA': $Control_PvIA, 
		'IAvIA': $Control_IAvIA}
	for name in nodes_control.keys():
		nodes_control[name].visible = name == mode
		buttons[name].pressed = name == mode
	

func _update_agents_infos(names):
	Utils.AGENTS_NAME = names
	$Control_PvIA/OptionButton.clear()
	for name in names:
		$Control_PvIA/OptionButton.add_item(name)
