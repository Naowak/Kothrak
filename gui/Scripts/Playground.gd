extends Spatial

var Character = preload("res://Scenes/Character.tscn")

var CellStage = {0: preload("res://Scenes/Cells/CellSize1.tscn"),
				1: preload("res://Scenes/Cells/CellSize2.tscn"),
				2: preload("res://Scenes/Cells/CellSize3.tscn"),
				3: preload("res://Scenes/Cells/CellSize4.tscn"),
				4: preload("res://Scenes/Cells/CellSize5.tscn")}

var grid = {}
var players = []


# Create all players and place them to their correct location
func instance_players(players_location):
	for player_id in players_location.keys():
		var coord = players_location[player_id]
		var player = Character.instance()
		player.init(coord[0], coord[1], Utils.players_colors[player_id])
		add_child(player)
		players += [player]


# Create all cells
func instance_map():
	var border_stage = 0
	_instance_cell(CellStage[1], 0, 0, 1)
	for radius in range(Utils.RAY+1):
		if radius <= Utils.RAY_ARENA:
			# Playable cells
			_instance_circle(radius, 1, true)
		else:
			# Border cells
			_instance_circle(radius, border_stage, false)
			border_stage += 1


# Instance one circle around the playble cells
func _instance_circle(radius, stage, playable):
	# Initialize directions
	var directions = [[0, -1], [-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1]]
	var q = 1*radius
	var r = 0*radius
	# For each direction, create nb radius cells
	for dir in directions:
		# warning-ignore:unused_variable
		for i in range(radius):
			_instance_cell(CellStage[stage], q, r, stage, playable)
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


func grow_up(cell):
	var stage = cell.stage + 1
	if stage >= Utils.MAX_STAGE:
		return

	cell.queue_free()
	_instance_cell(CellStage[stage+1], cell.q, cell.r, stage)
	
# Return the distance between two coordonates
#func distance_coord(c1, c2):
#	var q1 = c1[0]
#	var r1 = c1[1]
#	var q2 = c2[0]
#	var r2 = c2[1]
#	return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2
