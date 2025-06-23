# Global state for communication between pygame_part and main (pygame and CustomTkinter)
game_actions = {
    "offer_draw": False,
    "surrender": False,
    "show_analysis_bar": False,
    "end_game": False,
    "draw_accept":False,
    "white_surrender": False,
    "red_surrender" : False,
    "draw" : False,
    "Eval": 0,
    "game_saving" : False
    #check if the game should be finished due to no figures present
    #"finish_game - no figures": False,
}


