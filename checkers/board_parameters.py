import pygame

# Initialize constants with default values
_width = 800
_height = 800
_rows, _cols = 8, 8
_square_size = _width // _cols

# Define getter functions
def get_width():
    return _width

def get_height():
    return _height

def get_rows():
    return _rows

def get_cols():
    return _cols

def get_square_size():
    return _square_size

# Define setter functions
def set_width(width):
    global _width, _square_size
    _width = width
    _square_size = _width // _cols

def set_height(height):
    global _height
    _height = height

def set_rows_cols(rows, cols):
    global _rows, _cols, _square_size
    _rows, _cols = rows, cols
    _square_size = _width // _cols

