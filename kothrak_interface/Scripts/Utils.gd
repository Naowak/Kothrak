extends Node

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
