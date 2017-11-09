import pygame, sys, math, time

class TSPGraphics:
    def __init__(self, width = 1200, height=900):
        self.width, self.height = width, height
        self.nodesize = 12
        pygame.init()
        self.surface = pygame.display.set_mode((width, height))

    def draw_frame(self, tsp_problem, iteration):
        self.surface.fill((255,255,255))
        pygame.display.set_caption("Travelling Salesman Problem - Iteration {}".format(iteration))
        for data in tsp_problem.dataset:
            center_x = math.floor(data[0] *(self.width-20))+10
            center_y = math.floor(data[1] *(self.height-20))+10

            pygame.draw.rect(self.surface, (255,0,0), pygame.Rect(center_x-self.nodesize/2, center_y-self.nodesize/2, self.nodesize, self.nodesize))

        for i, node in enumerate(tsp_problem.nodes):
            center_x = math.floor(node.weights[0] *(self.width-20))+10
            center_y = math.floor(node.weights[1] *(self.height-20))+10
            pygame.draw.circle(self.surface, (0,0,255), (center_x, center_y), int(self.nodesize/4))


            next_x = math.floor(tsp_problem.nodes[(i+1)%len(tsp_problem.nodes)].weights[0] *(self.width-20))+10
            next_y = math.floor(tsp_problem.nodes[(i+1)%len(tsp_problem.nodes)].weights[1] *(self.height-20))+10

            pygame.draw.line(self.surface, (0, 255, 255), (center_x, center_y), (next_x, next_y), 1)


        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
    def wait(self):
        display = True
        while display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: display = False
            time.sleep(0.1)
