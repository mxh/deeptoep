import os, sys
import pygame
from pygame.locals import *

from ui_helpers import *

from game import *

class ToepUIMain:
    def __init__(self, width=1280, height=1024):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))

    def set_game(self, game):
        self.game = game

        self.update_sprites()

    def update_sprites(self):
        self.card_sprites = {}
        self.card_sprites_group = pygame.sprite.Group()
        for player_idx in range(0, len(self.game.players)):
            self.card_sprites[player_idx] = {}
            self.card_sprites[player_idx]['hand'] = {}
            self.card_sprites[player_idx]['table'] = {}

            # hand
            for card_idx in range(0, len(self.game.players[player_idx].hand)):
                self.card_sprites[player_idx]['hand'][card_idx] = {'front': CardSprite(self.game.players[player_idx].hand[card_idx]), 'back': CardBackSprite()}
                self.card_sprites_group.add(self.card_sprites[player_idx]['hand'][card_idx]['front'], self.card_sprites[player_idx]['hand'][card_idx]['back']

            # table
            for card_idx in range(0, len(self.game.players[player_idx].table)):
                self.card_sprites[player_idx]['table'][card_idx] = {'front': CardSprite(self.game.players[player_idx].table[card_idx]), 'back': CardBackSprite()}
                self.card_sprites_group.add(self.card_sprites[player_idx]['table'][card_idx]['front'], self.card_sprites[player_idx]['table'][card_idx]['back']

    def main_loop(self):
        self.load_sprites()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            draw_hand(self.game.phase.current_player, True)
            for player_idx in range(0, len(self.game.players)):
                #open_hand = player_idx == self.game.phase.current_player
                #draw_hand(player_idx, open_hand)
                #draw_table(player_idx)
                pass

            pygame.display.flip()

    def draw_hand(self, player_idx):


    def load_sprites(self):
        self.back = CardBackSprite()
        self.back_sprites = pygame.sprite.RenderPlain((self.back))

        self.card_sprites = {}
        for suit in suits:
            for value in values:
                self.card_sprites[(value, suit)] = pygame.sprite.RenderPlain((CardSprite((value, suit))))

class CardBackSprite(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image('./img/png/back.png', -1)

class CardSprite(pygame.sprite.Sprite):
    def __init__(self, card):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image('./img/png/{0}o{1}.png'.format(card[0], card[1]), -1)

if __name__=="__main__":
    toep_game = ToepGame()
    toep_ui = ToepUIMain()
    toep_ui.set_game(toep_game)
    toep_ui.main_loop()
