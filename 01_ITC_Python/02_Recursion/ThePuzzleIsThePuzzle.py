"""
This is my implementation of the main_template exercise
Author: Jeremy Bensoussan
"""
import math

# Create constants indicating the piece types
MIDDLE = 0
SIDE = 1
CORNER = 2
UNDEF = -1

# Create constants indicating the index for each side
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Piece(object):
    """This class holds puzzle pieces """
    cube_id = 0
    n_rotations = 0
    # initial slices has the original orientation of the piece, the one for which n_rotations = 0
    initial_slices = []
    # slices has the updated version of the piece, according to n_rotations
    slices = []
    piece_type = UNDEF

    def __init__(self, cube_id, slices):
        """Initializer function for the Piece class"""
        self.cube_id = cube_id
        self.initial_slices = slices
        self.slices = slices

        # the values used for these constants is equal to the number of 0 found in the slices
        self.piece_type = slices.count(0)

    def rotate(self, n_rotations):
        """Applies clockwise rotations and returns """

        # Make sure the number of rotations is between 0 and 3
        n_rotations = n_rotations % 4

        self.n_rotations = n_rotations % 4

        # n_shift determines by how many indices we need to increment each index (always keep it modulo 4)
        n_shift = (4 - n_rotations) % 4

        # update the current orientation of the piece
        self.slices = [self.initial_slices[(UP + n_shift) % 4], self.initial_slices[(RIGHT + n_shift) % 4],
                       self.initial_slices[(DOWN + n_shift) % 4], self.initial_slices[(LEFT + n_shift) % 4]]

    def has_border(self, direction):
        """Returns a boolean indicating if the piece has a border in the parameter direction"""
        return self.slices[direction] == 0

    def can_connect(self, piece_2, side):
        """If piece_2 is compatible with the piece self on the provided side,
        this function will return the number of rotations
        e.g. if the side provided is UP, this function compares the UP value of self to the
         DOWN value of piece_2"""

        # return False if the side we're trying to add a piece to is a border
        if self.slices[side] == 0:
            return False

        # we use the modulo operand in order to compare opposite sides
        # if the opposite sides are not equal, return False
        if self.slices[side] != piece_2.slices[(side+2) % 4]:
            return False

        # if the opposite sides are equal, we have more checking to do before we can give a green light:
        # if self is a CORNER, we need to make sure piece_2 is a SIDE piece
        if CORNER == self.piece_type:
            if SIDE != piece_2.piece_type:
                return False

        # if both pieces are a SIDE or a CORNER, we need to make sure they have one border in the same direction
        if self.piece_type in (CORNER, SIDE) and piece_2.piece_type in (CORNER, SIDE):
            if not (self.slices[UP] == piece_2.slices[UP] == 0 or self.slices[RIGHT] == piece_2.slices[RIGHT] == 0
               or self.slices[DOWN] == piece_2.slices[DOWN] == 0 or self.slices[LEFT] == piece_2.slices[LEFT] == 0):
                return False

        # if we made it so far return True
        return True

    def piece_tuple_string(self):
        """Formats a piece according to the required answer (cube_id, n_rotations)"""
        return '%d,%d' % (self.cube_id, self.n_rotations)


class Puzzle(object):
    """This class holds a list of tuples pieces in order and provides methods allowing to check
    if a piece will fit in a given place.
    We assume all puzzles provided will be square of side > 2 pieces"""
    piece_list = []
    total_pieces = 0
    puzzle_side = 0

    def __init__(self, total_pieces):
        """Initializer function for the Piece class"""
        self.total_pieces = total_pieces
        self.piece_list = [None] * total_pieces
        self.puzzle_side = int(math.sqrt(total_pieces))

    def copy(self):
        """creates a new puzzle as a copy of self (by value)"""
        copied_puzzle = Puzzle(self.total_pieces)
        copied_puzzle.piece_list = self.piece_list.copy()
        return copied_puzzle

    def replicate(self, new_puzzle):
        """this functions allows to keep the same puzzle object reference but have
        the variables of the new_puzzle"""
        self.piece_list = new_puzzle.piece_list
        self.total_pieces = new_puzzle.total_pieces
        self.puzzle_side = new_puzzle.puzzle_side

    def is_adjacent_to_border(self, position, direction):
        """This function takes a position and a direction as parameters
        and returns True if position is adjacent to a border in the provided direction"""
        if UP == direction:
            # determine border according to position value and puzzle size e.g. position < 10 in a 10x10 puzzle
            return position < self.puzzle_side

        if RIGHT == direction:
            # right border is determined by the modulo - e.g. 9, 19, 29... for a 10x10 puzzle
            return position % self.puzzle_side == self.puzzle_side - 1

        if DOWN == direction:
            # down border is touched if position >= side * (side-1) e.g. >= 90 for a 10x10 puzzle
            return position >= self.puzzle_side * (self.puzzle_side - 1)

        if LEFT == direction:
            # left border is touched if the modulo to the side is 0, e.g. 0, 10, 20, 30... for a 10x10 puzzle
            return position % self.puzzle_side == 0

        return False

    def get_adjacent_pieces(self, position):
        """This function returns a list of 4 elements.
        Each element is either a reference to the piece in one of the 4 directions, or None
        if that direction is a border, or if the puzzle is not yet filled in that position.
        We used the same direction principles as the exercise: [UP, RIGHT, DOWN, LEFT]"""
        # first initialize the adjacent pieces table to all None values
        adjacent_pieces = [None] * 4

        # calculate index to find corresponding values in each direction
        if not self.is_adjacent_to_border(position, UP):
            adjacent_pieces[UP] = self.piece_list[position-self.puzzle_side]
        if not self.is_adjacent_to_border(position, RIGHT):
            adjacent_pieces[RIGHT] = self.piece_list[position+1]
        if not self.is_adjacent_to_border(position, DOWN):
            adjacent_pieces[DOWN] = self.piece_list[position+self.puzzle_side]
        if not self.is_adjacent_to_border(position, LEFT):
            adjacent_pieces[LEFT] = self.piece_list[position - 1]

        return adjacent_pieces

    def piece_fits_position(self, piece, position):
        """This function checks if a piece fits in the provided position
        Returns True if the addition was possible, False otherwise"""
        # return False if a piece exists already in this position
        if self.piece_list[position] is not None:
            return False

        # return False when we're trying to add a non-border piece
        # to a border location or conversely
        if piece.has_border(UP) != self.is_adjacent_to_border(position, UP):
            return False
        if piece.has_border(RIGHT) != self.is_adjacent_to_border(position, RIGHT):
            return False
        if piece.has_border(DOWN) != self.is_adjacent_to_border(position, DOWN):
            return False
        if piece.has_border(LEFT) != self.is_adjacent_to_border(position, LEFT):
            return False

        # get adjacent pieces and test each one to confirm it is safe to add
        adjacent_pieces = self.get_adjacent_pieces(position)

        # in each direction, if a piece exists return False if it is not compatible with the piece we are trying to add
        for direction in range(0, 4):
            if adjacent_pieces[direction] is not None and not piece.can_connect(adjacent_pieces[direction], direction):
                return False

        # if we made it so far return True
        return True

    def add_piece(self, piece, position):
        """This function adds a puzzle piece if it fits in the provided position
        Returns True if the addition was possible, False otherwise"""
        if self.piece_fits_position(piece, position):
            self.piece_list[position] = piece
            return True

        return False

    def puzzle_result_string(self):
        """Returns the puzzle string according to the required format,
        i.e. (cube_id, n_rotations) tuples, separated by semi colons.
        Returns an empty string if the puzzle is not yet completed"""

        # Use list comprehension and join in order to generate the string as per the requirement
        return '; '.join([piece.piece_tuple_string() if piece is not None else 'None' for piece in self.piece_list])

    def puzzle_display_board(self):
        """This function displays the puzzle in a readable manner"""
        result = ''
        for i in range(0, self.puzzle_side):
            result += ' | '.join(['  %s  ' % ('0'+str(self.piece_list[i*self.puzzle_side+j].slices[UP]))[-2:]
                                  if self.piece_list[i*self.puzzle_side+j] is not None
                                  else '  xx  ' for j in range(0, self.puzzle_side)]) + '\n'
            result += ' | '.join(['%s  %s' % (('0'+str(self.piece_list[i*self.puzzle_side+j].slices[LEFT]))[-2:],
                                              ('0' + str(self.piece_list[i * self.puzzle_side + j].slices[RIGHT]))[-2:])
                                  if self.piece_list[i*self.puzzle_side+j] is not None
                                  else 'xx  xx' for j in range(0, self.puzzle_side)]) + '\n'
            result += ' | '.join(['  %s  ' % ('0'+str(self.piece_list[i*self.puzzle_side+j].slices[DOWN]))[-2:]
                                  if self.piece_list[i*self.puzzle_side+j] is not None
                                  else '  xx  ' for j in range(0, self.puzzle_side)]) + '\n'
            if i < self.puzzle_side-1:
                result += '-+-'.join(['------'] * self.puzzle_side) + '\n'

        return result


def read_pieces_from_string(puzzle_as_string):
    """This function reads all pieces from the provided string and creates a list of pieces"""
    # initialize empty array of unsorted pieces
    result = []

    # loop through the split the string of pieces in order to create the corresponding objects
    for piece_description in puzzle_as_string.split(';'):
        current_piece_id = int(piece_description[0:piece_description.find(',')])
        current_piece_slice = list(map(int, piece_description[piece_description.find(',') + 2:-1].split(',')))

        result.append(Piece(current_piece_id, current_piece_slice))

    return result


def read_pieces_from_input_file():
    """This function reads the pieces file and passes it as a string to read_puzzle_from_string"""
    # read the pieces from the provided text file
    file = open('puzzle.txt', 'r')
    input_pieces = file.read()

    return read_pieces_from_string(input_pieces)


def possible_placements(puzzle_board, shuffled_pieces, position):
    """This function returns a list of all the possible placements for
    a given position. Also, our algo doesn't allow to place a piece with no
    neighbour, except for corners."""
    possible_placements = []

    for piece in shuffled_pieces:
        for rotation in range(0, 4):
            piece.rotate(rotation)
            if puzzle_board.piece_fits_position(piece, position):
                possible_placements.append((piece, rotation))

    return possible_placements


def resolve_puzzle(puzzle_board, shuffled_pieces, next_position=0, verbose=False):
    """This function resolves the puzzle by looking at each position from the top left
    and trying each piece and each rotation until a valid placement is found.
    If multiple placements are possible, it will create a duplicate of the board
    and recursively call itself. Lastly, when a solution is found, it is returned
    to its parent which will return it until it goes back to the main loop."""

    # loop through all positions in the puzzle board and fills it.
    # recursively branch the board when multiple piece placements are possible for a position
    for position in range(next_position, len(puzzle_board.piece_list)):

        # possible pieces stores the possible piece along with the necessary rotation
        # this way, if two rotations of the same piece are possible, each will be considered
        possible_pieces = possible_placements(puzzle_board, shuffled_pieces, position)
        # if there is no possible piece, we are in a dead branch. return False and move on to the next possible piece
        if len(possible_pieces) == 0:
            if verbose:
                print('position %d: 0 possibility' % position)
            return False
        # if only one piece is possible, drop it and keep in the same branch
        elif len(possible_pieces) == 1:
            next_piece = possible_pieces[0][0]
            next_piece.rotate(possible_pieces[0][1])
            if verbose:
                print('position %d: try 1 / 1 possibility \t\t - %s' % (position, next_piece.slices))
            puzzle_board.add_piece(next_piece, position)
            shuffled_pieces.remove(next_piece)
        # this is where the magic happens, when multiple solutions are possible, create branches
        else:
            # loop through each possible (piece, rotation) couple
            for piece_rotation_tuple in possible_pieces:
                next_piece = piece_rotation_tuple[0]
                next_piece.rotate(piece_rotation_tuple[1])
                if verbose:
                    print('position %d: try %d / %d possibilities \t - %s' % (position,
                                                                  possible_pieces.index(piece_rotation_tuple)+1,
                                                                  len(possible_pieces), next_piece.slices))
                # copy the puzzle_board and add the piece to the branched_puzzle_board
                branched_puzzle_board = puzzle_board.copy()
                branched_shuffled_pieces = shuffled_pieces.copy()
                branched_puzzle_board.add_piece(next_piece, position)

                # remove the piece that was just added from the remaining shuffled pieces
                branched_shuffled_pieces.remove(next_piece)

                # call the function recursively, if it reaches a dead end, false will be returned and
                # it will come back to this loop and try the next piece.
                # if it finds a puzzle solution, True will be returned to the parent branch until it reaches
                # the main branch
                if resolve_puzzle(branched_puzzle_board, branched_shuffled_pieces, position+1, verbose):
                    # I've implemented a puzzle.replicate function so that the initial board will contain the solution
                    puzzle_board.replicate(branched_puzzle_board)

                    # the shuffled pieces is just a list that can be emptied and filled with new values
                    shuffled_pieces.clear()
                    shuffled_pieces.extend(branched_shuffled_pieces)
                    return True

            # return false here if none of the possible pieces reached a possible solution
            return False

    # finally, if the position loop is over, it means we have placed all the pieces successfully.
    # we can call it a success and return true
    return True


def main():
    """Main function that reads the pieces from the input file, creates a puzzle and solves it
    lastly, it displays the results as a board and as a string as required """

    # turn this variable to True to print the algorithm step by step
    verbose = True

    # first create a list of pieces according to the file. The pieces are not yet in order, hence the name
    shuffled_pieces = read_pieces_from_input_file()

    # then initialize a puzzle object which will receive pieces only if they fit
    puzzle_board = Puzzle(len(shuffled_pieces))

    # this function will resolve the puzzle, and the solution will be available in the puzzle_board variable
    resolve_puzzle(puzzle_board, shuffled_pieces, verbose=verbose)

    print(puzzle_board.puzzle_display_board())
    print('The result formated as requested in the exercise:')
    print(puzzle_board.puzzle_result_string())


if __name__ == '__main__':
    main()
