extends Spatial

var gid = null


func _ready():
	$HTTPRequest.new_game()


# Update the game with new data from server
func _update(data):
	# New game
	if gid == null:
		_new_game(data)
	# Update current game
	else:
		pass



# Create map and character instances, retrieve some informations
func _new_game(data):
	# Informations
	gid = data['gid']
	# Instance map
	$Map.instance_map(data['state']['cells_stage'])
	# Instance player
	$Map.instance_player(data['state']['current_player'], 'player')
	$Map.instance_player(data['state']['opponents'], 'opponent')



				
#
#func find_move(data):
#	# Find self move
#	var newg = data['state']['current_player']
#	var oldg = $Map.g
	


