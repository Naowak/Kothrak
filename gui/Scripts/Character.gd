extends Spatial

var q
var r
var color


func _ready():
	scale = Vector3(0.5, 0.5, 0.5)
	translation.y = 1
	_color_black_parts()


# Set position in 3D space and change color
func init(_q, _r, _color):
	q = _q
	r = _r
	color = _color
	
	translation.x = q * Utils.TRANS_RIGHT.x + r * Utils.TRANS_DOWNRIGHT.x
	translation.z = r * Utils.TRANS_DOWNRIGHT.y
	_color_body(color)


# Color the whole body (exept eyes and mouth) to color
func _color_body(_color):
	$Arm_left.set_surface_material(0, Utils.materials[_color])
	$Arm_right.set_surface_material(0, Utils.materials[_color])
	$Foot_left.set_surface_material(0, Utils.materials[_color])
	$Foot_right.set_surface_material(0, Utils.materials[_color])
	$Head.set_surface_material(0, Utils.materials[_color])
	$Body.set_surface_material(0, Utils.materials[_color])


# Color eyes and mouth to black
func _color_black_parts():
	$Eye_left.set_surface_material(0, Utils.materials['black'])
	$Eye_right.set_surface_material(0, Utils.materials['black'])
	$Mouth.set_surface_material(0, Utils.materials['black'])
