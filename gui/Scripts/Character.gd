extends Spatial



func _ready():
	_color_black_parts()
	color_body('red')


func _color_black_parts():
	$Eye_left.set_surface_material(0, Utils.materials['black'])
	$Eye_right.set_surface_material(0, Utils.materials['black'])
	$Mouth.set_surface_material(0, Utils.materials['black'])

func color_body(color):
	$Arm_left.set_surface_material(0, Utils.materials[color])
	$Arm_right.set_surface_material(0, Utils.materials[color])
	$Foot_left.set_surface_material(0, Utils.materials[color])
	$Foot_right.set_surface_material(0, Utils.materials[color])
	$Head.set_surface_material(0, Utils.materials[color])
	$Body.set_surface_material(0, Utils.materials[color])
	
