import pygame

#colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
OCHRE = (204, 119, 34)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)
ORANGE = (255, 102, 0)

#Crown image
pygame.init()
CROWN = pygame.transform.scale(pygame.image.load('checkers/assets/crown.png'), (44, 25))