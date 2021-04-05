extends Spatial

var gid = null


func _ready():
	$HTTPRequest.new_game()


# Update the game with new data from server
func _update(data):
	# New game
	if data['status'] == 'new_game':
		_new_game(data)
	# Update current game
	else:
		print('Unknown behaviour')



# Create map and character instances, retrieve some informations
func _new_game(data):
	# Informations & settings
	gid = data['gid']
	Utils.update_settings(data)
	# Instance map
	$Playground.instance_map()
	# Instance player
	$Playground.instance_players(data['players_location'])



				
#
#func find_move(data):
#	# Find self move
#	var newg = data['state']['current_player']
#	var oldg = $Map.g
	


