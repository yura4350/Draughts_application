import pygame
import threading
from checkers.game import Game
from minimax.algorithm import minimax
from checkers import board_parameters
from checkers.constants import *
import shared_state

# Using getter functions
WIDTH = board_parameters.get_width()
HEIGHT = board_parameters.get_height()
ROWS, COLS = board_parameters.get_rows(), board_parameters.get_cols()
SQUARE_SIZE = board_parameters.get_square_size()


#setting FPS
FPS = 60

#get the position of the mouse
def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // board_parameters.get_square_size()
    col = x // board_parameters.get_square_size()
    return row, col

def run_pygame(mode, color=None, difficulty=None):
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Checkers')
    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)


    while run:
        clock.tick(FPS)

        #set length of analysis by pre-installed bar
        bar_length_of_analysis = 4

        # Getting the evaluation of the position
        Eval, position = minimax(game.get_board(), bar_length_of_analysis, True, game, -100, 100)
        shared_state.game_actions["Eval"] = round(Eval, 2)

        #check the mode of play:
        if mode == "player_vs_computer":

            #length of analysis - how many moves ahead does the algorithm analyse
            length_of_analysis = int(difficulty)+3

            #create the conditions for user playing for different colors
            if color == "Red":
                if game.turn == WHITE:
                    value, new_board = minimax(game.get_board(), length_of_analysis, True, game, -100, 100)
                    game.ai_move(new_board)

            else:
                if game.turn == RED:
                    value, new_board = minimax(game.get_board(), length_of_analysis, False, game, -100, 100)
                    game.ai_move(new_board)

                    # Check for game actions
            if shared_state.game_actions["offer_draw"]:
                #Implementing draw offer logic so that for more than 1.0 for whtie and -1.0 for red threshold in evaluation computer accepted the draw
                Eval, position = minimax(game.get_board(), length_of_analysis, True, game, -100, 100)


                if color == "Red" and Eval <= -1.0:
                    shared_state.game_actions["draw_accept"] = True
                    run = False

                elif color == "White" and Eval >= 1.0:
                    shared_state.game_actions["draw_accept"] = True
                    run = False

                # Handle draw offer logic
                shared_state.game_actions["offer_draw"] = False  # Reset flag after handling

            if shared_state.game_actions["surrender"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["end_game"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["game_saving"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop
        else:

            if shared_state.game_actions["white_surrender"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["red_surrender"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["draw"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["end_game"]:
                # Handle surrender logic
                run = False  # Example action: stop the game loop

            if shared_state.game_actions["game_saving"]:
                # Handle surrender logic
                run = False  # Example action: stop the game lo


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

        game.update()

    pygame.quit()

# Starting pygame loop in a new thread
def start_pygame_thread(mode = None, color=None, difficulty=None, board_size=None):
    print(mode, color, difficulty, board_size)

    #setting parameters of the board according to user's choice
    if mode == "player_vs_computer":
        print(1, 1)
        if board_size == "8x8":
            board_parameters.set_rows_cols(8, 8)
        elif board_size == "10x10":
            board_parameters.set_rows_cols(10, 10)
        else:
            board_parameters.set_rows_cols(12, 12)
        pygame_thread = threading.Thread(target=run_pygame, args=(mode, color, difficulty))
        pygame_thread.daemon = True
        pygame_thread.start()

    elif mode == "player_vs_player":
        if board_size == "8x8":
            board_parameters.set_rows_cols(8, 8)
        elif board_size == "10x10":
            board_parameters.set_rows_cols(10, 10)
        else:
            board_parameters.set_rows_cols(12, 12)
        pygame_thread = threading.Thread(target=run_pygame, args=(mode, color, difficulty))
        pygame_thread.daemon = True
        pygame_thread.start()


