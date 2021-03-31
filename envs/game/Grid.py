class Cell:
    def __init__(self, _q, _r, _stage):
        self.q = _q
        self.r = _r
        self.stage = _stage
        

MAX_STAGE = 4
DIR_COORDS = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]


class Grid:

    def __init__(self, app, ray):
        self.ray = ray
        self.grid = []
        self._create_grid()


    def _create_grid(self):
        """Create the map
        """
        nb_cell = self.ray + 1
        q = 0
        r = -self.ray

        # create negative r coord lines
        for i in range(self.ray):
            self.grid += [list()]
            q, r = self._create_one_line(nb_cell, q, r)
            nb_cell += 1
            r += 1
            q = -self.ray - r

        # create positive r coord lines
        for i in range(self.ray + 1):
            self.grid += [list()]
            q, r = self._create_one_line(nb_cell, q, r)
            r += 1
            q = -self.ray
            nb_cell -= 1


    def _create_one_line(self, nb_cell, q, r):
        for _ in range(nb_cell):
            self.grid[r + self.ray] += [Cell(q, r, 1)]
            q += 1
        return q, r


    def get_cell_from_coord(self, q, r):
        for line in self.grid:
            for cell in line:
                if cell.q == q and cell.r == r:
                    return cell


    def get_all_cells(self):
        cells = []
        for line in self.grid:
            cells += line
        return cells
    

    def get_neighbors(self, cell, ray=1, with_none=False):

        if cell is None:
            return []

        neighbors_coord = [(cell.q + c[0], cell.r + c[1])
                           for c in get_neighbors_rel_coord(ray)]
        neighbors = [self.get_cell_from_coord(q, r) 
                        for q, r in neighbors_coord]

        if not with_none: 
            neighbors = [c for c in neighbors if c is not None]

        return neighbors


def distance(coord_1, coord_2):
    return (abs(coord_1[0] - coord_2[0]) 
            + abs(coord_1[0] + coord_1[1] - coord_2[0] - coord_2[1]) 
            + abs(coord_1[1] - coord_2[1])) / 2


def get_neighbors_rel_coord(ray, dir_coords=DIR_COORDS):
    rel_coords = [(0, 0)] + dir_coords
    queue = [c for c in dir_coords]

    while len(queue) > 0:
        c1 = queue.pop(0)

        for c2 in dir_coords:
            new_coord = (c1[0] + c2[0], c1[1] + c2[1])
            
            if distance(new_coord, (0, 0)) <= ray and \
                    new_coord not in rel_coords:

                queue += [new_coord]
                rel_coords += [new_coord]

    return list(rel_coords)
