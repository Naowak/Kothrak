from kothrak.envs.game.Utils import GRID_RAY, POS_CENTER, MV_DR, MV_R
from kothrak.envs.game.Cell import Cell


DIR_COORDS = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]

def distance(coord_1, coord_2):
    return (abs(coord_1[0] - coord_2[0])
            + abs(coord_1[0] + coord_1[1] - coord_2[0] - coord_2[1])
            + abs(coord_1[1] - coord_2[1])) / 2

def get_neighbors_rel_coord(ray=GRID_RAY, dir_coords = DIR_COORDS):
    rel_coords = {c for c in dir_coords}
    queue = [c for c in dir_coords]

    while len(queue) > 0:
        c1 = queue.pop(0)

        for c2 in dir_coords:
            new_coord = (c1[0] + c2[0], c1[1] + c2[1])
            
            if new_coord not in rel_coords and distance(new_coord, (0, 0)) <= ray:
                queue += [new_coord]
                rel_coords.add(new_coord)

    return list(rel_coords)



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
    
    def get_neighbors(self, cell, ray=1, with_none=False):

        neighbors_coord = [(cell.q + c[0], cell.r + c[1])
                           for c in get_neighbors_rel_coord(ray)]
        neighbors = [self.get_cell_from_coord(q, r) for q, r in neighbors_coord]

        if not with_none: 
            neighbors = [c for c in neighbors if c is not None]

        return neighbors
    
