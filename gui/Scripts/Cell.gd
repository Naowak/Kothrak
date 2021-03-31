extends Spatial

# signal
signal cell_clicked

# Geometrical data
const CIRCLE_RAY = 1
const SPACE_BETWEEN = 0
const DIST = sqrt(3)*CIRCLE_RAY
const RATIO = (DIST + SPACE_BETWEEN)/DIST
const TRANS_RIGHT = Vector2(DIST*RATIO, 0)
const TRANS_DOWNRIGHT = Vector2(DIST*RATIO/2, 3.0*CIRCLE_RAY*RATIO/2)

# Game const
const MAX_STAGE = 4 

# cell attributes
var q
var r
var color
var stage
#var character_on setget set_character


func _ready():
	pass


func init(_q, _r, _stage, _color):
	q = _q
	r = _r
	stage = _stage
	color = _color
#	character_on = null
	
	translation.x = q * TRANS_RIGHT.x + r * TRANS_DOWNRIGHT.x
	translation.z = r * TRANS_DOWNRIGHT.y
	change_material(color)
	
	if _color == 'white':
		# warning-ignore:return_value_discarded
		var playground_node = get_tree().get_root().get_node('Playground')
		connect("cell_clicked", playground_node, '_on_cell_clicked', [self])


func change_material(material_key):
	$Circle.set_surface_material(0, Utils.materials[material_key])
	

func grew():
	stage = stage + 1
	


#func set_character(new_character):
#	character_on = new_character
#
#	if new_character != null:
#		kind = 'blocked'
#	else:
#		kind = 'floor'


func _on_Area_input_event(_camera, event, _click_position, _click_normal, _shape_idx):
	# If the event is a mouse click
	if event is InputEventMouseButton and event.pressed:
		if color == "white" :
			# A different material is applied on each button
			if event.button_index == BUTTON_LEFT :
				emit_signal('cell_clicked')
