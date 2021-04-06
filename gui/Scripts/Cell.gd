extends Spatial

# signal
signal cell_clicked

# cell attributes
var q
var r
var color
var stage


func _ready():
	pass


# Set position in 3D space, change color and connect events
func init(_q, _r, _stage, _color):
	q = _q
	r = _r
	stage = _stage
	color = _color
#	character_on = null
	
	translation.x = q * Utils.TRANS_RIGHT.x + r * Utils.TRANS_DOWNRIGHT.x
	translation.z = r * Utils.TRANS_DOWNRIGHT.y
	_change_color(color)
	
	if _color == 'white':
		# warning-ignore:return_value_discarded
		var main_node = get_tree().get_root().get_node('Main')
		connect("cell_clicked", main_node, '_play', [self])


# Change the color of the cell, color must be a string in Utils.materials.keys()
func _change_color(color_name):
	$Circle.set_surface_material(0, Utils.materials[color_name])
	

# Emit signal cell_clicked when the cell is left-clicked
func _on_Area_input_event(_camera, event, _click_position, _click_normal, _shape_idx):
	# If the event is a mouse click
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == BUTTON_LEFT :
			emit_signal('cell_clicked')
