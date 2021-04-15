extends Spatial

var players_kind = []

var gid = -1
var player_id = -1
var possible_plays = []
var step = ''

var tid = -1

# usefull for play
var data_to_send = null

func _ready():
	# warning-ignore:return_value_discarded
	$Panel/Control_PvP/Button_newgame.connect('pressed', $HTTPRequest, 'request_new_session', ['PvP'])
	# warning-ignore:return_value_discarded
	$Panel/Control_PvIA/Button_newgame.connect('pressed', $HTTPRequest, 'request_new_session', ['PvIA'])
	# warning-ignore:return_value_discarded
	$Panel/Control_IAvIA/Button_newtrain.connect('pressed', $HTTPRequest, 'request_new_session', ['IAvIA'])

# Update the game with new data from server
func _update(data):
	print(data)
	# New game
	if data['status'] == 'new_game':
		_new_game_update(data)
	# Update current game
	elif data['status'] == 'playing':
		_playing_update(data)
	# Player won
	elif data['status'] == 'win':
		_win_update(data)
	# Player eliminated
	elif data['status'] == 'eliminated':
		step = 'game_over'
		_update_logs(gid, data['status'], player_id, step)
	
	# Status received for trainings
	# New training
	elif data['status'] == 'start':
		tid = data['tid']
		$HTTPRequest.request_watch_training(tid)
	# New game to watch
	elif data['status'] == 'watch':
		for infos in data['history']:
			_update(infos)
			yield(get_tree().create_timer(1.0), "timeout")
		$HTTPRequest.request_watch_training(tid)
	# ADD ELIF WHERE TRAIN IS OVER


# Create map and character instances, retrieve some informations
func _new_game_update(data):
	# Informations & settings
	if Utils.MODE in ['PvP', 'PvIA']:
		gid = data['gid']
	Utils.update_server_settings(data['settings'])
	# Instance map
	$Playground.instance_map()
	# Instance player
	$Playground.instance_players(data['players_location'])
	_init_players_kind()
	# Turn infos
	_playing_update(data)

# Update the game with informations received
func _playing_update(data):
	# Retrieve informations from server
	player_id = data['player_id']
	possible_plays = data['possible_plays']
	step = 'move'
	
	# If not new_game and last player was IA, make the play
	var prev_player_id = fposmod(player_id-1, Utils.NB_PLAYERS)
	if data['status'] != 'new_game' and players_kind[prev_player_id] == 'IA':
		# Make the move
		var abs_move = _to_absolute(data['move'], prev_player_id)
		$Playground.move(prev_player_id, $Playground.grid[abs_move[0]][abs_move[1]])
		# Make the build
		var abs_build = _to_absolute(data['build'], prev_player_id)
		$Playground.grow_up($Playground.grid[abs_build[0]][abs_build[1]])
	
	# Update panel text
	_update_logs(gid, data['status'], player_id, step)
	
	# If next player is IA, request server to play
	if Utils.MODE == 'PvIA' and players_kind[player_id] == 'IA':
		$HTTPRequest.request_watch(gid)
		

# Game over, current player won. 
# Make the last move if player_id is IA and update_text.
func _win_update(data):
	if players_kind[player_id] == 'IA':
		var abs_move = _to_absolute(data['move'], player_id)
		$Playground.move(player_id, $Playground.grid[abs_move[0]][abs_move[1]])
	step = 'game_over'
	_update_logs(gid, data['status'], player_id, step)
	

	
# Verify if play is correct and play it. 
# Send information to the server if turn is over.
# Called by event cell_clicked
func _make_play(cell):
	
	if players_kind[player_id] != 'Person':
		print("Error: It's not to a Person to play.")
		return
	
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
			data_to_send['move'] = _to_relative([cell.q, cell.r], player_id)
			$Playground.move(player_id, cell)
			possible_plays = play_remaining
			step = 'build'
			_update_logs(gid, 'playing', player_id, step)
			
			# If no build possible, the player reach the last stage, he won
			if possible_plays[0]['build'] == null:
				$HTTPRequest.request_human_play(gid, data_to_send)
				step = 'game_over'
				print('Game Over !')
	
	elif step == 'build':
		# Verify the build, play it 
		for play in possible_plays:
			var coord = play['build']
			if coord[0] == cell.q and coord[1] == cell.r:
				# cell present in possible plays				
				valid = true
				data_to_send['build'] = _to_relative(play['build'], player_id)
		if valid:
			$Playground.grow_up(cell)
			$HTTPRequest.request_human_play(gid, data_to_send)
			step = 'move'


func _init_players_kind():
	players_kind = []
	if Utils.MODE == 'PvP':
		for _i in range(Utils.NB_PLAYERS):
			players_kind += ['Person']
			
	elif Utils.MODE == 'PvIA':
		for _i in range(Utils.NB_PERSON):
			players_kind += ['Person']
		for _i in range(Utils.NB_AGENTS):
			players_kind += ['IA']
		players_kind.shuffle()
	
	elif Utils.MODE == 'IAvIA':
		for _i in range(Utils.NB_AGENTS):
			players_kind += ['IA']
	
	print(players_kind)


# Update logs corresponding to MODE
func _update_logs(_gid, _status, _player_id, _step):
	var path_node = 'Panel/Control_' + str(Utils.MODE) + '/Logs'
	var node = get_node(path_node)
	node.update_text(_gid, _status, _player_id, _step)


# Remove the player coordinates to coord
func _to_relative(coord, pid):
	return [coord[0] - $Playground.players[pid].q, 
		coord[1] - $Playground.players[pid].r]

# Add the player coordinates to coord
func _to_absolute(coord, pid):
	return [coord[0] + $Playground.players[pid].q,
		coord[1] + $Playground.players[pid].r]




