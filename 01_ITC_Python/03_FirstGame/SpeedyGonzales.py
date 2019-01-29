"""בס׳׳ד
This is my implementation of the SpeedyGonzales game
Author: Jeremy Bensoussan
"""
import sys
import pygame
import time
import os

pygame.init()

DISPLAY_WIDTH = 700
DISPLAY_HEIGHT = 300
# Frames Per Second
frame_per_second = 60
# number of frames when animating a loosing or winning speedy gonzales
reduced_frame_per_second = 8

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREY = (150, 150, 150)

LEFT_STEP_KEY = pygame.K_a
RIGHT_STEP_KEY = pygame.K_k
N_LAST_KEY_TIMES = 10
STEPS_TO_WIN = 80

last_key = None
last_key_time = 0
last_key_times = [0]
game_end = False
game_won = False
shown_high_scores = False

game_display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Speedy Gonzalessss!!!')
clock = pygame.time.Clock()

# Image Names
SPRITESHEET = 'SpeedySpriteSheet.png'
BACKGROUND = 'background.png'
HIGH_SCORES = 'score_file.txt'

# Sprites Constant Definition
STANDING_SPEEDY_SPRITE = (19, 32, 64, 50)
RUNNING_SPEEDY_STRIP = (19, 97, 64, 50)
RUNNING_SPEEDY_N_SPRITES = 5
SPEEDING_SPEEDY_SPRITE = (19, 316, 64, 50)
SPEEDING_SPEEDY_N_SPRITES = 4
DEAD_SPEEDY_SPRITE = (30, 1836, 52, 50)
DEAD_SPEEDY_N_SPRITES = 4
WIN_SPEEDY_SPRITE = (108, 815, 56, 50)
WIN_SPEEDY_N_SPRITES = 4

STANDING_SPEEDY_INDEX = 0
RUNNING_SPEEDY_INDEX = 1
SPEEDING_SPEEDY_INDEX = 2
DEAD_SPEEDY_INDEX = 3
WIN_SPEEDY_INDEX = 4

# Load Background Image
background_image = pygame.image.load(BACKGROUND)
BACKGROUND_WIDTH = 1400

highscore_font = pygame.font.SysFont("Arial", 16)


class SpriteSheet(object):
    """Object allowing to load sprites from a file containing a spritesheet"""
    def __init__(self, filename):
        """Constructor of the spritesheet object"""
        try:
            self.sheet = pygame.image.load(filename).convert()
        except pygame.error:
            print('Unable to load spritesheet image: %s' % filename)
            raise SystemExit

    # Load a specific image from a specific rectangle
    def image_at(self, rectangle, colorkey=None):
        """Loads image from x,y,x+offset,y+offset"""
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size)
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey is -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

        return image

    # Load a whole bunch of images and return them as a list
    def images_at(self, rects, colorkey=None):
        """Loads multiple images, supply a list of coordinates"""
        return [self.image_at(rect, colorkey) for rect in rects]

    # Load a whole strip of images
    def load_strip(self, rect, image_count, colorkey=None):
        """Loads a strip of images and returns them as a list"""
        tups = [(rect[0]+rect[2]*x, rect[1], rect[2], rect[3])
                for x in range(image_count)]
        return self.images_at(tups, colorkey)


def read_from_file_and_find_highscore(file_name):
    """Reads data from high score file"""
    high_score = sys.maxsize
    high_name = ''

    if os.path.exists(file_name):
        high_score_file = open(file_name, 'r')
        lines = high_score_file.readlines()
        high_score_file.close()

        for line in lines:
            name, score = line.strip().split(",")
            score = float(score)

            if score < high_score:
                high_score = score
                high_name = name

    else:
        high_score_file = open(file_name, 'w')
        high_score_file.close()

    return high_name, high_score


def write_to_file(file_name, your_name, points):
    """Exports high scores"""
    score_file = open(file_name, 'a')
    print(your_name + ",", points, file=score_file)
    score_file.close()


def show_top10(screen, file_name, new_score_name='', new_score_points=0):
    """Displays top 10 highscores"""
    bx = 480  # x-size of box
    by = 400  # y-size of box

    file = open(file_name, 'r')
    lines = file.readlines()

    all_score = []
    for line in lines:
        sep = line.index(',')
        name = line[:sep]
        score = float(line[sep + 1:-1])
        all_score.append((score, name))
    file.close()
    all_score.sort()  # sort from largest to smallest
    best = all_score[:10]  # top 10 values

    # make the presentation box
    # box = pygame.surface.Surface((bx, by))
    box = pygame.Surface((bx, by))
    box.fill(GREY)
    pygame.draw.rect(box, WHITE, (50, 12, bx - 100, 35), 0)
    pygame.draw.rect(box, WHITE, (50, by - 60, bx - 100, 42), 0)
    pygame.draw.rect(box, BLACK, (0, 0, bx, by), 1)
    txt_surf = highscore_font.render("HIGHSCORES", True, BLACK)  # headline
    txt_rect = txt_surf.get_rect(center=(bx // 2, 30))
    box.blit(txt_surf, txt_rect)
    txt_surf = highscore_font.render("Press ENTER to continue", True, BLACK)  # bottom line
    txt_rect = txt_surf.get_rect(center=(bx // 2, 360))
    box.blit(txt_surf, txt_rect)

    best_score_found = False

    # write the top-10 data to the box
    for i, entry in enumerate(best):
        text_color = BLACK
        if new_score_name == entry[1] and new_score_points == entry[0] and not best_score_found:
            best_score_found = True
            text_color = RED

        txt_surf = highscore_font.render((' ' + str(i + 1))[-2:] + '. ' + entry[1], True, text_color)
        txt_rect = txt_surf.get_rect()
        txt_rect.x = bx // 2-150
        txt_rect.y = 30 * i + 50
        box.blit(txt_surf, txt_rect)
        txt_surf = highscore_font.render(str(entry[0]), True, text_color)
        txt_rect = txt_surf.get_rect()
        txt_rect.x = bx // 2+80
        txt_rect.y = 30 * i + 50
        box.blit(txt_surf, txt_rect)

        screen.blit(box, (0, 0))
    pygame.display.flip()

    while True:  # wait for user to acknowledge and return
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return
        pygame.time.wait(20)


def enterbox(screen, txt):
    """Defines behaviour of input box"""
    def blink(dsplay):
        for color in [GREY, WHITE]:
            pygame.draw.circle(box, color, (bx // 2, int(by * 0.7)), 7, 0)
            dsplay.blit(box, (0, by // 2))
            pygame.display.flip()
            pygame.time.wait(300)

    def show_name(dsplay, score_name):
        pygame.draw.rect(box, WHITE, (50, 60, bx - 100, 20), 0)
        text_surf = highscore_font.render(score_name, True, BLACK)
        text_rect = text_surf.get_rect(center=(bx // 2, int(by * 0.7)))
        box.blit(text_surf, text_rect)
        dsplay.blit(box, (0, by // 2))
        pygame.display.flip()

    bx = 480
    by = 100

    # make box
    # box = pygame.surface.Surface((bx, by))
    box = pygame.Surface((bx, by))
    box.fill(GREY)
    pygame.draw.rect(box, BLACK, (0, 0, bx, by), 1)
    txt_surf = highscore_font.render(txt, True, BLACK)
    txt_rect = txt_surf.get_rect(center=(bx // 2, int(by * 0.3)))
    box.blit(txt_surf, txt_rect)

    name = ""
    show_name(screen, name)

    # the input-loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                inkey = event.key
                if inkey in [13, 271]:  # enter/return key
                    return name
                elif inkey == 8:  # backspace key
                    name = name[:-1]
                elif inkey <= 300:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT and 122 >= inkey >= 97:
                        inkey -= 32  # handles CAPITAL input
                    name += chr(inkey)

        if name == "":
            blink(screen)
        show_name(screen, name)


def highscore(screen, file_name, your_points):
    """Is called to enter a new highscore """
    high_name, high_score = read_from_file_and_find_highscore(file_name)
    your_name = ''

    if your_points < high_score:
        your_name = enterbox(screen, "YOU HAVE BEATEN THE HIGHSCORE - What is your name?")

    elif your_points == high_score:
        your_name = enterbox(screen, "YOU HAVE SAME AS HIGHSCORE - What is your name?")

    elif your_points > high_score:
        st1 = "Highscore is "
        st2 = " made by "
        st3 = "   What is your name?"
        txt = st1 + str(high_score) + st2 + high_name + st3
        your_name = enterbox(screen, txt)

    if your_name is None or len(your_name) == 0:
        return  # do not update the file unless a name is given

    write_to_file(file_name, your_name, your_points)
    show_top10(screen, file_name, your_name, your_points)
    return


def load_speedy_sprites():
    """Creates a list of generators for each sprite that may be used in game"""
    speedy_sprite_sheet = SpriteSheet(SPRITESHEET)
    generators = []

    standing_image = speedy_sprite_sheet.image_at(STANDING_SPEEDY_SPRITE, -1)
    running_images = speedy_sprite_sheet.load_strip(RUNNING_SPEEDY_STRIP, RUNNING_SPEEDY_N_SPRITES, colorkey=-1)
    speeding_images = speedy_sprite_sheet.load_strip(SPEEDING_SPEEDY_SPRITE, SPEEDING_SPEEDY_N_SPRITES, colorkey=-1)
    dead_images = speedy_sprite_sheet.load_strip(DEAD_SPEEDY_SPRITE, DEAD_SPEEDY_N_SPRITES, colorkey=-1)
    win_images = speedy_sprite_sheet.load_strip(WIN_SPEEDY_SPRITE, WIN_SPEEDY_N_SPRITES, colorkey=-1)

    generators.append(image_generator([standing_image]))
    generators.append(image_generator(running_images))
    generators.append(image_generator(speeding_images))
    generators.append(image_generator(dead_images))
    generators.append(image_generator(win_images))

    return generators


def image_generator(images):
    """This generator function returns the next image in a list constituting a sprite"""
    index = 0
    while True:
        yield images[index]
        index = (index + 1) % len(images)


def move_character_to_position(character_image_generator, x, y):
    """Increments sprite index and displays image at parameter coordinates"""
    time_diff = last_key_times[-1] - last_key_times[0]

    if game_end and not game_won:
        generator_index = DEAD_SPEEDY_INDEX
    elif game_end and game_won:
        generator_index = WIN_SPEEDY_INDEX
    elif time.time() - last_key_time > 0.08:
        generator_index = STANDING_SPEEDY_INDEX
    elif len(last_key_times) == N_LAST_KEY_TIMES and time_diff / N_LAST_KEY_TIMES < 0.065:
        generator_index = SPEEDING_SPEEDY_INDEX
    else:
        generator_index = RUNNING_SPEEDY_INDEX

    game_display.blit(next(character_image_generator[generator_index]), (x, y))


def message_display(text, x=0, y=0, font_size=20, color=BLACK, center=False):
    """Displays a text according to parameters"""
    font = pygame.font.SysFont('Comic Sans MS', font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = ((DISPLAY_WIDTH / 2), (DISPLAY_HEIGHT / 2))
    else:
        text_rect.x = x
        text_rect.y = y
    game_display.blit(text_surface, text_rect)


def init_game():
    """Resets the global variables before a new party is started"""
    global last_key, last_key_time, last_key_times, game_end, game_won, frame_per_second, shown_high_scores
    last_key = None
    last_key_time = 0
    last_key_times = [0]
    game_end = False
    game_won = False
    shown_high_scores = False
    frame_per_second = 60


def game_loop():
    """Where the main tasks occur"""
    global last_key, last_key_time, last_key_times, game_end, game_won, frame_per_second, shown_high_scores
    speedy_generators = load_speedy_sprites()

    speedy_y_relative_position = 0.6
    finish_relative_position = 0.85

    # This loop allows to play one game after the other
    while True:
        x = 0
        y = (DISPLAY_HEIGHT * speedy_y_relative_position)
        finish_line = (DISPLAY_WIDTH * finish_relative_position)
        x_change = (finish_line - x) / STEPS_TO_WIN
        n_keys = 0
        initial_time = None
        init_game()
        restart = False
        final_time = 0

        # Make sure the sprite's bottom pixel is at position y
        move_character_to_position(speedy_generators, x, y)

        # The actual game loop that executes the changes in display, as well as game over / won
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                if not game_end:
                    if event.type == pygame.KEYDOWN:
                        # User pressed one of the 2 valid keys
                        if event.key in (RIGHT_STEP_KEY, LEFT_STEP_KEY):
                            # Fail if same key was pressed twice
                            if last_key == event.key:
                                final_time = time.time()
                                game_end = True
                                game_won = False
                                frame_per_second = reduced_frame_per_second
                            # Else update the character position
                            else:
                                # Init timer if first move
                                if initial_time is None:
                                    initial_time = time.time()

                                last_key = event.key
                                last_key_time = time.time()
                                last_key_times.append(last_key_time)
                                if len(last_key_times) > N_LAST_KEY_TIMES:
                                    last_key_times.pop(0)
                                x += x_change
                                n_keys += 1

                        if x >= finish_line:
                            final_time = time.time()
                            print('finished in %f' % (final_time-initial_time))
                            frame_per_second = reduced_frame_per_second
                            game_end = True
                            game_won = True
                else:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and shown_high_scores:
                            restart = True
                            break

            if restart:
                break

            # Clean the screen from previous state
            game_display.fill(WHITE)

            # Scroll background according to Speedy's progression
            background_x_pos = -1 * (BACKGROUND_WIDTH - DISPLAY_WIDTH) * n_keys / STEPS_TO_WIN
            game_display.blit(background_image, (background_x_pos, 0))

            # updates time, and on screen message if game ended or won
            if initial_time is not None:
                message_display('step = %d' % int(n_keys / 2), 10, 35, 30, WHITE)
                if not game_end:
                    message_display('time = %.4f' % (time.time() - initial_time), 10, 0, 30, WHITE)
                else:
                    message_display('time = %.4f' % (final_time - initial_time), 10, 0, 30, WHITE)
                    if game_won:
                        message_display('You won!', center=True, font_size=90, color=WHITE)
                        if time.time() - final_time > 4 and not shown_high_scores:
                            show_highscores(final_time - initial_time)
                    else:
                        message_display('You lost!', center=True, font_size=90, color=WHITE)
                        if time.time() - final_time > 4 and not shown_high_scores:
                            show_highscores()
                    if shown_high_scores:
                        message_display('<space> to restart', 190, 220, font_size=40, color=WHITE)

            move_character_to_position(speedy_generators, x, y)

            # Update display
            pygame.display.update()
            clock.tick(frame_per_second)


def show_highscores(game_time=0.0):
    """Displays the list of highscores in a reduced window"""
    global shown_high_scores
    pygame.display.set_mode((480, 400))
    if game_time > 0:
        highscore(game_display, HIGH_SCORES, round(game_time, 4))
    else:
        show_top10(game_display, HIGH_SCORES)
    pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    shown_high_scores = True


def main():
    """The main function starts here"""
    if not os.path.exists(SPRITESHEET):
        print('Please make sure the file %s is in the current working directory' % SPRITESHEET)
        sys.exit(0)

    if not os.path.exists(BACKGROUND):
        print('Please make sure the file %s is in the current working directory' % BACKGROUND)
        sys.exit(0)

    game_loop()


if __name__ == '__main__':
    main()
