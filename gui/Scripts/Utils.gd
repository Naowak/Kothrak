extends Node

# Geometrical data
const CIRCLE_RAY = 1
const SPACE_BETWEEN = 0
const DIST = sqrt(3)*CIRCLE_RAY
const RATIO = (DIST + SPACE_BETWEEN)/DIST
const TRANS_RIGHT = Vector2(DIST*RATIO, 0)
const TRANS_DOWNRIGHT = Vector2(DIST*RATIO/2, 3.0*CIRCLE_RAY*RATIO/2)

# Server parameters
var MAX_STAGE = 4
var RAY_ARENA = 2
var NB_PLAYERS = 2
var AGENTS_NAME = []

# Interface parameters
var MODE = 'PvP'
var NB_PERSON = 2
var NB_AGENTS = 0

var LENGTH_BORDER = 3
var RAY = RAY_ARENA + LENGTH_BORDER

# Materials dict
var materials = {'white': "e6cab8", 
				'black': '352f2b',
				'blue': '2876df',
				'red': 'df4828',
				'green': '79cc2b',
				'grey': 'c6beba'
				}

var players_colors = ['blue', 'red', 'green', 'grey']


func _ready():
	_init_materials()


# Replace all hex code in materials by their SpatialMaterial instance
func _init_materials():
	for key in materials.keys():
		var color = materials[key]
		var mat = SpatialMaterial.new()
		mat.albedo_color = Color(color)
		materials[key] = mat

# Replace settings values 
func update_server_settings(settings):
	for name in settings.keys():
		if name in 'MAX_STAGE':
			MAX_STAGE = settings[name]
		elif name == 'RAY':
			RAY_ARENA = settings[name]
			RAY = RAY_ARENA + LENGTH_BORDER
		elif name == 'NB_PLAYERS':
			NB_PLAYERS = settings[name]
