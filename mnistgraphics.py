import pygame, sys, math, time

class MNISTGraphics:
    def __init__(self, width = 1000, height=1000):
        self.width, self.height = width, height
        pygame.init()
        #pygame.font.init()
        self.surface = pygame.display.set_mode((width, height))
        self.colors = [(0,0,102),(51,0,102), (0,153,51), (0,255,255), (255,0,51),(204,102,51), (255,51,255),(255,255,51), (255,0,0),(102,0,0)]
        #self.font = pygame.font.Font(pygame.font.get_default_font(), 30)

    def draw_frame(self, problem):
        self.surface.fill((255,255,255))
        for node in problem.nodes:
            x = node.x
            y = node.y
            dim_width = node.nodes_per_dim
            number = node.get_number()

            startx = math.floor((x*self.width)/dim_width)
            starty = math.floor((y*self.height)/dim_width)
            pygame.draw.rect(self.surface, self.colors[number], pygame.Rect(startx, starty, math.floor(self.width/dim_width), math.floor(self.height/dim_width)))

            #drawtext = myfont.render('{}'.format(number), False, (0, 0, 0))
            #self.surface.blit(drawtext,(startx,starty))

        pygame.display.flip()
    def wait(self):
        display = True
        while display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: display = False
            time.sleep(0.1)
