from kothrak.envs.game.Utils import GRID_RAY, POS_CENTER, MV_DR, MV_R
from kothrak.envs.game.Cell import Cell

class Grid:

    def __init__(self, app, ray=GRID_RAY):
        self.ray = ray
        self.app = app
        self.grid = []
        self.create_grid()

    def create_grid(self):
        """Create the map"""

        def create_one_cell(self, q, r):
            pos_x = POS_CENTER[0] + MV_R[0]*q + MV_DR[0]*r
            pos_y = POS_CENTER[1] + MV_DR[1]*r
            return Cell(pos_x, pos_y, q, r, self.app)

        def create_one_line(self, nb_cell, q, r):
            for _ in range(nb_cell):
                c = create_one_cell(self, q, r)
                self.grid[r + self.ray] += [c]
                q += 1
            return q, r

        nb_cell = self.ray + 1
        q = 0
        r = -self.ray
        for i in range(self.ray):
            self.grid += [list()]
            q, r = create_one_line(self, nb_cell, q, r)
            nb_cell += 1
            r += 1
            q = -self.ray - r
        for i in range(self.ray + 1):
            self.grid += [list()]
            q, r = create_one_line(self, nb_cell, q, r)
            r += 1
            q = -self.ray
            nb_cell -= 1

    def get_cell_from_coord(self, q, r):
        for line in self.grid:
            for cell in line:
                if cell.q == q and cell.r == r:
                    return cell

    def get_cell_from_pos(self, x, y):
        cells = []
        for r in reversed(range(-self.ray, self.ray+1)):
            for line in self.grid:
                for c in line:
                    if c.r == r:
                        cells += [c]

        for c in cells:
            if c.is_pos_in_cell(x, y):
                return c

    def get_all_cells(self):
        cells = []
        for line in self.grid:
            for cell in line:
                cells += [cell]
        return cells
    
    def delete(self):
        for line in self.grid:
            for cell in line:
                cell.delete()
