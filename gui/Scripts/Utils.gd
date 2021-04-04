extends Node

# Geometrical data
const CIRCLE_RAY = 1
const SPACE_BETWEEN = 0
const DIST = sqrt(3)*CIRCLE_RAY
const RATIO = (DIST + SPACE_BETWEEN)/DIST
const TRANS_RIGHT = Vector2(DIST*RATIO, 0)
const TRANS_DOWNRIGHT = Vector2(DIST*RATIO/2, 3.0*CIRCLE_RAY*RATIO/2)

# Game const
const LENGTH_BORDER = 3
const RAY_ARENA = 2
const RAY = LENGTH_BORDER + RAY_ARENA
const MAX_STAGE = 4 

# Materials dict
var materials = {'white': "e6cab8", 
				'black': '352f2b',
				'blue': '2876df',
				'red': 'df4828',
				'green': '79cc2b',
				'grey': 'c6beba'
				} 

func _ready():
	_init_materials()

func _init_materials():
	for key in materials.keys():
		var color = materials[key]
		var mat = SpatialMaterial.new()
		mat.albedo_color = Color(color)
		materials[key] = mat
