extends Panel

# Called when the node enters the scene tree for the first time.
func _ready():
	# warning-ignore:return_value_discarded
	$Button_1v1.connect('pressed', self, '_on_control_change', ['1v1'])
	# warning-ignore:return_value_discarded
	$Button_1vIA.connect('pressed', self, '_on_control_change', ['1vIA'])
	# warning-ignore:return_value_discarded
	$Button_IAvIA.connect('pressed', self, '_on_control_change', ['IAvIA'])


func _on_control_change(mode):
	var buttons = {'1v1': $Button_1v1, '1vIA': $Button_1vIA, 
		'IAvIA': $Button_IAvIA}
	var nodes_control = {'1v1': $Control_1v1, '1vIA': $Control_1vIA, 
		'IAvIA': $Control_IAvIA}
	for name in nodes_control.keys():
		nodes_control[name].visible = name == mode
		buttons[name].pressed = name == mode
	
