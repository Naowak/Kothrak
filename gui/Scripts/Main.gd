extends Spatial

var gid = -1
var player_id = -1
var possible_plays = []
var step = ''

# usefull for play
var data_to_send = null

func _ready():
	print('hello')
	$HTTPRequest.new_game()


# Update the game with new data from server
func _update(data):
	print(data)
	# New game
	if data['status'] == 'new_game':
		_new_game(data)
	# Update current game
	elif data['status'] == 'playing':
		_playing(data)
	elif data['status'] == 'win' or data['status'] == 'eliminated':
		print('End of server game')


# Create map and character instances, retrieve some informations
func _new_game(data):
	# Informations & settings
	gid = data['gid']
	Utils.update_settings(data['settings'])
	# Instance map
	$Playground.instance_map()
	# Instance player
	$Playground.instance_players(data['players_location'])
	# Turn infos
	player_id = data['player_id']
	possible_plays = data['possible_plays']
	step = 'move'

# Update the game with informations received
func _playing(data):
	player_id = data['player_id']
	possible_plays = data['possible_plays']
	step = 'move'
	

# Verify if play is correct and play it. 
# Send information to the server if turn is over.
# Called by event cell_clicked
func _play(cell):
	var valid = false
	
	if step == 'move':
		data_to_send = {'move': null, 'build': null}
		
		# Verify move in possible plays and update possible plays
		var play_remaining = []
		for play in possible_plays:
			if play['move'][0] == cell.q and play['move'][1] == cell.r:
				# cell present in possible plays
				play_remaining += [play]
				valid = true
				
		# Make the move and send infos to server
		if valid:
			data_to_send['move'] = _to_relative([cell.q, cell.r])
			$Playground.move(player_id, cell)
			possible_plays = play_remaining
			step = 'build'
			
			# If no build possible, the player reach the last stage, he won
			if possible_plays[0]['build'] == null:
				$HTTPRequest.play(gid, data_to_send)
				step = 'game_over'
				print('Game Over !')
	
	elif step == 'build':
		# Verify the build, play it 
		for play in possible_plays:
			var coord = play['build']
			if coord[0] == cell.q and coord[1] == cell.r:
				# cell present in possible plays				
				valid = true
				data_to_send['build'] = _to_relative(play['build'])
		if valid:
			$Playground.grow_up(cell)
			$HTTPRequest.play(gid, data_to_send)
			step = 'move'
			

# Remove the player coordinates to coord
func _to_relative(coord):
	return [coord[0] - $Playground.players[player_id].q, 
		coord[1] - $Playground.players[player_id].r]




