SIZE_MAP = 5
NB_PLAYERS = 2
PLAYERS_COLOR = ["\033[0;37;40m", "\033[0;40;31m", "\033[0;40;32m", "\033[0;40;34m", "\033[0;40;33m"]
MAX_CELL_HEIGHT = 3

class Kothrak() :
    
    def __init__(self) :
        self.map = [[0 for j in range(SIZE_MAP)] for i in range(SIZE_MAP)]
        self.players = [[-1, -1] for _ in range(NB_PLAYERS)]
        self.turn = -1
        
    def ask_build(self, player) :
        correct_play = False
        while not correct_play :
            try :
                y = int(input("Build : Numéro de ligne : ")) - 1
                x = int(input("Build : Numéro de colonne : ")) - 1
            except ValueError :
                print("Invalid Value.")
                continue
            if not self.is_in_map(x, y) :
                print('Cell not in map.')
                continue
            if not self.is_cell_free(x, y) :
                print("Cell not free.")
                continue
            if self.get_cell_height(x, y) >= MAX_CELL_HEIGHT :
                print("Cannot build on this cell")
                continue
            player_y, player_x = self.players[player-1]
            if self.is_player_on_map(player) :
                if not self.is_cell_in_range(player_x, player_y, x, y, 1) :
                    print('Location too far.')
                    continue            
            else :
                print("Player {} not in map.".format(player))
            correct_play = True
        return x, y
    
    def ask_movement(self, player) :
        correct_play = False
        while not correct_play :
            try :
                y = int(input("Move : Numéro de ligne : ")) - 1
                x = int(input("Move : Numéro de colonne : ")) - 1
            except ValueError :
                print("Invalid Value.")
                continue
            if not self.is_in_map(x, y) :
                print('Cell not in map.')
                continue
            if not self.is_cell_free(x, y) :
                print("Cell already taken.")
                continue
            old_y, old_x = self.players[player-1]
            if self.is_player_on_map(player) :
                if not self.is_cell_in_range(old_x, old_y, x, y, 1) :
                    print('Location too far.')
                    continue
                if self.is_cell_too_high(old_x, old_y, x, y) :
                    print("Can not climb that high.")
                    continue
            correct_play = True
        return x, y

    def is_in_map(self, x, y) :
        for v in [x, y] :
            if v < 0 or v >= SIZE_MAP :
                return False
        return True
    
    def is_cell_free(self, x, y) :
        return not [y, x] in self.players
    
    def is_cell_in_range(self, xfrom, yfrom, x, y, range) :
        for v, vfrom in [(x, xfrom), (y, yfrom)] :
            if abs(v - vfrom) > range :
                return False
        return True
    
    def is_cell_too_high(self, xfrom, yfrom, x, y) :
        return self.map[y][x] - self.map[yfrom][xfrom] > 1
    
    def is_player_on_map(self, player) :
        return not -1 in self.players[player-1]
    
    def get_cell_height(self, x, y) :
        return self.map[y][x]
            
    def is_game_over(self) :
        for j in range(SIZE_MAP) :
            for i in range(SIZE_MAP) :
                if self.map[j][i] == 3 and [j, i] in self.players :
                    return self.players.index([j, i]) + 1
        return False
    
    def run(self) :
        player = 0
        game_over = False
        while(not game_over) :
            self.turn += 1
            player = player%2 + 1
            print("\nJoueur {}".format(player))
            print(self)
            move_x, move_y = self.ask_movement(player)
            self.players[player - 1] = [move_y, move_x]
            game_over = self.is_game_over()
            if game_over :
                continue
                
            if self.turn >= NB_PLAYERS :
                print(self)
                print("\nJoueur {}".format(player))
                build_x, build_y = self.ask_build(player)
                self.map[build_y][build_x] += 1

        print(self)
        print("Game Over !")
        print("Le joueur {} a gagné la partie !".format(game_over))
    
    def __str__(self) :
        string = PLAYERS_COLOR[0]
        string += "  | " + " ".join(["{}".format(i+1) for i in range(SIZE_MAP)]) + ' \n'
        string += "-" * (4 + 2*SIZE_MAP)
        for j in range(SIZE_MAP) :
            string += '\n{} | '.format(j+1)
            for i in range(SIZE_MAP) :
                color = PLAYERS_COLOR[0]
                if [j, i] in self.players :
                    color = PLAYERS_COLOR[self.players.index([j, i]) + 1]
                string += "{}".format(color)
                string += "{}".format(self.map[j][i])
                string += "{} ".format(PLAYERS_COLOR[0])
        return string  


if __name__ == '__main__':
    koth = Kothrak()
    koth.run()

    