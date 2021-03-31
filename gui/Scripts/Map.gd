extends Spatial

const LENGTH_BORDER = 3
const RAY_ARENA = 2
const RAY = LENGTH_BORDER + RAY_ARENA

var CellSize0 = preload("res://Scenes/Cells/CellSize1.tscn")
var CellSize1 = preload("res://Scenes/Cells/CellSize2.tscn")
var CellSize2 = preload("res://Scenes/Cells/CellSize3.tscn")
var CellSize3 = preload("res://Scenes/Cells/CellSize4.tscn")
var CellSize4 = preload("res://Scenes/Cells/CellSize5.tscn")


var grid = {}
var cells = []
var last_mouse_position = Vector2(-1, -1)


func _ready():
	generate_grid()
	instance_map(grid)


## HANDLE GRID GENERATION ##
func _generate_one_gridline(line_size, r):
	var color = ''
	var half = line_size / 2 if (line_size / 2.0)  == (line_size / 2) else (line_size / 2 + 1)
	var q = -RAY -r if r <= 0 else -RAY
	for i in range(half):
		if distance_coord(q, r, 0, 0) > RAY_ARENA:
			color = 'black'
		else:
			color = 'white'
		_add_instance_to_grid(color, q, r)
		if line_size / 2.0 == line_size / 2 or i + 1 != half :
			_add_instance_to_grid(color, line_size - 2*i + q - 1, r)
		q += 1

func generate_grid():
	var nb_cell = RAY + 1
	for r in range(-RAY, 0) :
		_generate_one_gridline(nb_cell, r)
		nb_cell += 1
	for r in range(RAY + 1) :
		_generate_one_gridline(nb_cell, r)
		nb_cell -= 1


## HANDLE GRID INSTANCIATION
func _add_instance_to_grid(instance, q, r):
	if not q in grid.keys():
		grid[q] = {}
	grid[q][r] = instance
	
func _instance_cell(cell_type, q, r, stage, color):
	var cell = cell_type.instance()
	add_child(cell)	
	cell.init(q, r, stage, color)
	_add_instance_to_grid(cell, q, r)
	if color == "white":
		cells += [cell]

func instance_map(new_grid):
	grid = new_grid
	for q in grid.keys():
		for r in grid[q].keys():
			var color = grid[q][r]
			if color == 'white':
				_instance_cell(CellSize1, q, r, 1, color)
			elif color == 'black':
				var dist = distance_coord(q, r, 0, 0) - RAY_ARENA
				var choices = {1: CellSize0, 2: CellSize1, 3:CellSize2}
				_instance_cell(choices[dist], q, r, dist-1, color)
				
func grew(cell):
	var stage = cell.stage + 1
	if stage >= 4:
		return
		
	cell.queue_free()
	var choices = {2: CellSize2, 3:CellSize3, 4:CellSize4}
	_instance_cell(choices[stage], cell.q, cell.r, stage, 'white')
	
	
## USEFULL FUNCTIONS
func distance_coord(q1, r1, q2, r2):
	return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

#func clear():
#	for c in cells:
#		c.change_material('white')
