import pygame
from checkers import board_parameters
from checkers.constants import *

# Using getter functions
WIDTH = board_parameters.get_width()
HEIGHT = board_parameters.get_height()
ROWS, COLS = board_parameters.get_rows(), board_parameters.get_cols()
SQUARE_SIZE = board_parameters.get_square_size()

class Piece:
    PADDING = 15
    OUTLINE = 2

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.SQUARE_SIZE = board_parameters.get_square_size()
        self.color = color
        self.king = False
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = self.SQUARE_SIZE * self.col + self.SQUARE_SIZE // 2
        self.y = self.SQUARE_SIZE * self.row + self.SQUARE_SIZE // 2

    #def make_king(self):
        #self.king = True

    def draw(self, win):
        radius = self.SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(win, GREY, (self.x, self.y), radius + self.OUTLINE)
        pygame.draw.circle(win, self.color, (self.x, self.y), radius)
        # if self.king:
           # win.blit(CROWN, (self.x - CROWN.get_width() // 2, self.y - CROWN.get_height() // 2))

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()

    def __repr__(self):
        return str(self.color)

class King(Piece):
    PADDING = 15
    OUTLINE = 2
    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.king = True  # This ensures that the piece is initialized as a King

    # No need to redefine calc_pos, draw, move, or __repr__ methods
    def draw(self, win):
        # Calling the draw method from the Piece class
        super().draw(win)
        # Additional King-specific drawing logic
        # Drawing a Crown
        win.blit(CROWN, (self.x - CROWN.get_width() // 2, self.y - CROWN.get_height() // 2))