extends Spatial

var CellStage = {0: preload("res://Scenes/Cells/CellSize1.tscn"),
				1: preload("res://Scenes/Cells/CellSize2.tscn"),
				2: preload("res://Scenes/Cells/CellSize3.tscn"),
				3: preload("res://Scenes/Cells/CellSize4.tscn"),
				4: preload("res://Scenes/Cells/CellSize5.tscn")}

const LENGTH_BORDER = 3
const RAY_ARENA = 2
const RAY = LENGTH_BORDER + RAY_ARENA

var grid = {}


# Create all playable cells corresponding to new_grid
func instance_map(new_grid):
	for q in new_grid.keys():
		for r in new_grid[q].keys():
			var stage = int(round(new_grid[q][r] * 3))
			_instance_cell(CellStage[stage], q, r, stage)


# Create all non playable cells 
func instance_border():
	var stage = 0
	for radius in range(RAY_ARENA+1, RAY+1):
		_instance_circle_border(radius, stage)
		stage += 1

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
		




	
				
#func grew(cell):
#	var stage = cell.stage + 1
#	if stage >= 4:
#		return
#
#	cell.queue_free()
#	var choices = {2: CellSize2, 3:CellSize3, 4:CellSize4}
#	_instance_cell(choices[stage], cell.q, cell.r, stage, 'white')
	
# Return the distance between two coordonates
#func distance_coord(c1, c2):
#	var q1 = c1[0]
#	var r1 = c1[1]
#	var q2 = c2[0]
#	var r2 = c2[1]
#	return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

#func clear():
#	for c in cells:
#		c.change_material('white')

#			var color = grid[q][r]
#			if color == 'white':
#				_instance_cell(CellSize1, q, r, 1, color)
#			elif color == 'black':
#				var dist = distance_coord(q, r, 0, 0) - RAY_ARENA
#				var choices = {1: CellSize0, 2: CellSize1, 3:CellSize2}
#				_instance_cell(choices[dist], q, r, dist-1, color)
