extends Spatial

var Character = preload("res://Scenes/Character.tscn")

var CellStage = {0: preload("res://Scenes/Cells/CellSize1.tscn"),
				1: preload("res://Scenes/Cells/CellSize2.tscn"),
				2: preload("res://Scenes/Cells/CellSize3.tscn"),
				3: preload("res://Scenes/Cells/CellSize4.tscn"),
				4: preload("res://Scenes/Cells/CellSize5.tscn")}

var grid = {}
var player = null
var opponent = null


# Create all cells corresponding to new_grid
func instance_map(new_grid):
	# Playable cells
	for q in new_grid.keys():
		for r in new_grid[q].keys():
			var stage = int(new_grid[q][r] * 3) + 1
			_instance_cell(CellStage[stage], q, r, stage)
	# Non playable cells
	var stage = 0
	for radius in range(Utils.RAY_ARENA+1, Utils.RAY+1):
		_instance_circle_border(radius, stage)
		stage += 1


func instance_player(data, faction):
	var coord = _retrieve_player_location(data)
	if faction == 'player':
		player = Character.instance()
		player.init(coord[0], coord[1], 'blue')
		add_child(player)
	elif faction == 'opponent':
		opponent = Character.instance()
		opponent.init(coord[0], coord[1], 'red')
		add_child(opponent)


# Instance one circle around the playble cells
func _instance_circle_border(radius, stage):
	# Initialize directions
	var directions = [[0, -1], [-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1]]
	var q = 1*radius
	var r = 0*radius
	# For each direction, create nb radius cells
	for dir in directions:
		# warning-ignore:unused_variable
		for i in range(radius):
			_instance_cell(CellStage[stage], q, r, stage, false)
			q += dir[0]
			r += dir[1]
	

# Instanciate one cell. If playable, add it to grid
func _instance_cell(cell_type, q, r, stage, playable=true):
	# Create one cell instance	
	var cell = cell_type.instance()
	add_child(cell)
	
	# Playable cell
	if playable:
		cell.init(q, r, stage, 'white')
		if not q in grid.keys():
			grid[q] = {}
		grid[q][r] = cell
		
	# Not a playable cell, but a border one
	else:
		cell.init(q, r, stage, 'black')
		

# Retrieve player location from a dictionnary
func _retrieve_player_location(data):
	for q in data.keys():
		for r in data[q].keys():
			if data[q][r] == 1:
				return [q, r]


#func grow_up(cell):
#	var stage = cell.stage + 1
#	if stage >= Utils.MAX_STAGE:
#		return
#
#	cell.queue_free()
#	_instance_cell(CellStage[stage+1], cell.q, cell.r, stage)
	
# Return the distance between two coordonates
#func distance_coord(c1, c2):
#	var q1 = c1[0]
#	var r1 = c1[1]
#	var q2 = c2[0]
#	var r2 = c2[1]
#	return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2
