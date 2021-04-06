extends RichTextLabel


func _ready():
	pass
	
func update_text(gid, status, player_id, step):
	var text = null
	text = '[center]Game id : ' + str(gid) + '[/center]\n'
	if status == 'new_game' or status == 'playing':
		text += '[center]' + step + '[/center]\n\n'
		for i in range(Utils.NB_PLAYERS):
			var color = Utils.players_colors[i]
			if i == player_id:
				text += '[color=' + color + ']Player ' + str(i) + ' *[/color]\n'
			else:
				text += '[color=' + color + ']Player ' + str(i) + '[/color]\n'
	
	elif status == 'win':
		var color = Utils.players_colors[player_id]
		text += '\n[color='+color+']Player '+str(player_id)+' won[/color]'
	
	elif status == 'eliminted':
		var color = Utils.players_colors[player_id]
		text += '\n[color='+color+']Player '+str(player_id)+' is eliminated[/color]'
		
	bbcode_text = text
			


