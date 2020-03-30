import pygame
import sys
from pygame.locals import *
import vectors_module
import math
import numpy as np
import random
import dill
import copy


pygame.init()

fps = 100

displayWidth = 1200

game_width = 576
game_height = 624

centerX = math.floor(game_width /2)

initialX = 100
initialY = game_height / 2

white = (255, 255, 255)
black = (0,0,0)
blue = (0,0,255)
red = (255,0,0)
green = (0,255,0)

hole_size = 80
gap = 300


display = pygame.display.set_mode((displayWidth, game_height))
pygame.display.set_caption('Flappy Bird')
clock = pygame.time.Clock()
myFont = pygame.font.SysFont('monospace', 20)
myFont2 = pygame.font.SysFont('monospace', 50)

number_birds = 100

redBirdMidImg = pygame.image.load('sprites/redbird-midflap.png').convert_alpha()
redBirdDownImg = pygame.image.load('sprites/redbird-downflap.png').convert_alpha()
redBirdUpImg = pygame.image.load('sprites/redbird-upflap.png').convert_alpha()


blueBirdMidImg = pygame.image.load('sprites/bluebird-midflap.png').convert_alpha()
blueBirdDownImg = pygame.image.load('sprites/bluebird-downflap.png').convert_alpha()
blueBirdUpImg = pygame.image.load('sprites/bluebird-upflap.png').convert_alpha()


yellowBirdMidImg = pygame.image.load('sprites/yellowbird-midflap.png').convert_alpha()
yellowBirdDownImg = pygame.image.load('sprites/yellowbird-downflap.png').convert_alpha()
yellowBirdUpImg = pygame.image.load('sprites/yellowbird-upflap.png').convert_alpha()

backgroundDayImg = pygame.image.load('sprites/background-day.png').convert_alpha()
backgroundNightImg = pygame.image.load('sprites/background-night.png').convert_alpha()

baseImg = pygame.image.load('sprites/base.png')

gameOver = pygame.image.load('sprites/gameover.png')

pipeGreenImg = pygame.image.load('sprites/pipe-green.png')
pipeRedImg = pygame.image.load('sprites/pipe-red.png')

soundObj = 'audio/Aakash Gandhi - Viking (Cinematic Background Music).mp3'
pygame.mixer.init()
pygame.mixer.music.load(soundObj)


birdColor = ['blue', 'red', 'yellow']
redBird = [redBirdUpImg, redBirdMidImg, redBirdDownImg]
blueBird = [blueBirdUpImg, blueBirdMidImg, blueBirdDownImg]
yellowBird = [yellowBirdUpImg, yellowBirdMidImg, yellowBirdDownImg]

birdWidth = redBirdMidImg.get_rect().size[0]
birdHeight = redBirdMidImg.get_rect().size[1]
obstacle_width = pipeGreenImg.get_rect().size[0]
obstacle_height = pipeGreenImg.get_rect().size[1]
backgroundWidth = backgroundDayImg.get_rect().size[0]
backgroundHeight = backgroundDayImg.get_rect().size[1]
baseWidth = baseImg.get_rect().size[0]
baseHeight = baseImg.get_rect().size[1]

number_obstacles = math.floor(game_width / (gap + obstacle_width)) + 1

brainStructure = (11, 1)

pipeSpeed = 5
birdAcceleration = 0.7
birdVelocity = 8
downfallSpeed = 5


def main():

    background = random.choice([backgroundDayImg, backgroundNightImg])

    # pygame.mixer.music.play(-1)

    # population = Population(number_birds, brainStructure)

    obstacles = [Obstacle(x=game_width + gap * i) for i in range(number_obstacles)]

    with open('fittest_brain.pkl', 'rb') as f:
        bird = dill.load(f)

    obstaclesPassed = 0

    while True:

        display.blit(background, (0, 0))
        display.blit(background, (backgroundWidth, 0))

        for i in range(number_obstacles):
            obstacles[i].move()
            obstacles[i].draw()

        # for i in range(population.size):
        #     if population.members[i].state == 'alive':
        #         if obstacles[0].x + obstacle_width <= population.members[i].position.x and population.members[i].position.x < obstacles[1].x - gap + pipeSpeed:
        #                 obstaclesPassed += 1
        #         break

        if obstacles[0].x + obstacle_width <= bird.position.x and bird.position.x < obstacles[1].x - gap + pipeSpeed:
            obstaclesPassed += 1

        numberObstaclesPassed = myFont2.render('%s' %(obstaclesPassed), 10, white)
        display.blit(numberObstaclesPassed, (centerX,30))

        obstacles_states = [obstacles[i].state for i in range(number_obstacles)]

        if 'dead' in obstacles_states:
            obstacles.pop(0)
            obstacles.append(Obstacle(x=obstacles[-1].x + gap + obstacle_width))

        display.blit(baseImg, (0, backgroundHeight))
        display.blit(baseImg, (baseWidth, backgroundHeight))

        display.fill(black, (game_width, 0, displayWidth - game_width, game_height))

        bird.move(obstacles)
        bird.draw()

        # for j in range(number_birds):
        #     population.members[j].move(obstacles)
        #     population.members[j].draw()

        # for j in range(number_birds):
        #     if population.members[j].state == 'alive':
        #         population.members[j].drawBrain()
        #         break

        # population.check_state()

        # generation = myFont.render('Generation: %s' %(population.generation), 1, white)
        # birdsAlive = myFont.render('Birds alive: %s' %(population.number_alive), 1, white)

        # if population.generation == 1:
        #     fittestBrain = myFont.render('Highest fitness: 0', 1, white)
        # else:
        #     fittestBrain = myFont.render('Highest fitness: %s' %(population.fittest_brain.fitness), 1, white)
        # display.blit(generation, (displayWidth - 300,20))
        # display.blit(birdsAlive, (displayWidth - 300,45))
        # display.blit(fittestBrain, (displayWidth - 300,70))

        # if population.state == 'dead':
        #     population.evolve(mutation_rate=0.15, keep='on', save='on')
        #     print('This is generation %s' % (population.generation))
        #     obstacles = [Obstacle(x=game_width + gap * i) for i in range(number_obstacles)]
            # obstaclesPassed = 0

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            # elif event.type == KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         population.state = 'dead'
            #         population.evolve(mutation_rate=0.15, keep='on', save='on')
            #         obstacles = [Obstacle(x=game_width + gap * i) for i in range(number_obstacles)]
            #         obstaclesPassed = 0

        pygame.display.flip()
        clock.tick(fps)


def estimate_angle(obj1, x, y):
    dx = x - obj1.x
    dy = y - obj1.y
    return math.atan2(dy, dx)


def estimate_distance(obj1, x, y):
    dx = x - obj1.x
    dy = y - obj1.y
    return math.sqrt(dx ** 2 + dy ** 2)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


class Obstacle(object):

    def __init__(self, x=None):
        if x == None:
            self.x = game_width
        else:
            self.x = x
        self.y = np.random.uniform(backgroundHeight - obstacle_height, obstacle_height)
        self.state = 'alive'
        self.rect1 = pygame.Rect((self.x, 0, obstacle_width, self.y))
        self.rect2 = pygame.Rect((self.x, self.y + hole_size, obstacle_width, game_height - hole_size - self.y))

    def draw(self):
        if self.state == 'alive' and self.x < game_width:
            rotatedPipe = pygame.transform.rotate(pipeGreenImg, 180)
            display.blit(rotatedPipe, (self.x, self.y - obstacle_height))
            display.blit(pipeGreenImg, (self.x, self.y + hole_size))

    def move(self):
        if self.x + obstacle_width >= 0:
            self.x = self.x - pipeSpeed
            self.rect1 = pygame.Rect((self.x, 0, obstacle_width, self.y))
            self.rect2 = pygame.Rect((self.x, self.y + hole_size, obstacle_width, game_height - hole_size - self.y))
        else:
            self.state = 'dead'


class Brain(object):

    def __init__(self, structure, radius=25, activation_function='sigmoid', body=None):
        self.structure = structure
        self.number_of_layers = len(self.structure)
        self.number_of_transitions = self.number_of_layers - 1
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights = None
        self.weights_initializer()
        self.activation_function = activation_function
        self.fitness = 0
        self.state = 'alive'
        self.output = None
        self.data = None
        self.position = vectors_module.vector_create(initialX, initialY)
        self.velocity = vectors_module.vector_create(0, 0)
        self.acceleration = vectors_module.vector_create(0, birdAcceleration)
        self.color = random.choice(birdColor)

    def predict(self, data):
        """
        If the input is a simple python list of observations, we transform it to numpy array type. If it is already a numpy array type, there will be no change.
        """
        length = len(data)
        """
        We check if the size of the data is equal to the number of input neurons
        """
        assert length == self.structure[0], 'ERROR: the length of the input list is not equal to the number of input neurons'

        data = np.reshape(data, (length, 1)).astype(float)
        # print(type(data))

        """
        We loop over all the transitions between the layers of our brain
        """
        for i in range(self.number_of_transitions):
            if self.activation_function == 'sigmoid':
                data = self.sigmoid(np.dot(self.weights[i], data) + self.biases[i])
            elif self.activation_function == 'ReLU':
                data = self.ReLU(np.dot(self.weights[i], data) + self.biases[i])
            elif self.activation_function == 'tanh':
                data = self.tanh(np.dot(self.weights[i], data) + self.biases[i])
        return data

    def weights_initializer(self):
        """
        The below code initializes the weights based on the number of newurons in each respective layer
        """
        self.weights = [np.random.normal(0, 1 / np.sqrt(x), (x, y)) for x, y in list(zip(self.structure[1:], self.structure[:-1]))]

    def save_as(self, filename):
        """
        Allows us to save a trained Brain object to file for later use
        """
        assert type(filename) == str, 'ERROR: filename should be type str'
        if '.pkl' in filename:
            with open(filename, 'wb') as f:
                dill.dump(self, f)
        else:
            with open(filename + '.pkl', 'wb') as f:
                dill.dump(self, f)

    def drawBrain(self):
        size = math.floor(game_height / self.structure[0])
        x = 10
        radius = math.floor((size -2*x) / 2)

        positions = [(game_width + 50, math.floor(size *(i+1) - (radius+x))) for i in range(self.structure[0])]

        outputs = [self.weights[0][0][i] * self.data[i] for i in range(self.structure[0])]

        neuronColors = [int(translate(self.data[i],min(self.data), max(self.data), 105,255)) for i in range(self.structure[0])]

        thicknessOutput = [int(translate(outputs[i], min(outputs), max(outputs), 1, 6)) for i in range(self.structure[0])]


        for i in range(self.structure[0]):
            pygame.draw.circle(display, (neuronColors[i],neuronColors[i],neuronColors[i]) , positions[i], radius, 0)
            if self.weights[0][0][i] >0:
                pygame.draw.line(display, blue, (positions[i][0] + radius, positions[i][1]), (game_width + 550 - radius-5,math.floor(game_height/2)), thicknessOutput[i])
            else:
                pygame.draw.line(display, red, (positions[i][0] + radius, positions[i][1]), (game_width + 550 - radius-5,math.floor(game_height/2)), thicknessOutput[i])


        if self.output >= 0.5:
            pygame.draw.circle(display, white, (game_width + 550,math.floor(game_height/2)), radius +5,0)
        elif self.output < 0.5:
            pygame.draw.circle(display, white, (game_width + 550,math.floor(game_height/2)), radius +5,1)


    def draw(self, color=white):

        if self.state == 'alive':
            if self.color == 'red':
                display.blit(random.choice(redBird), (self.position.x, self.position.y))
            elif self.color == 'blue':
                display.blit(random.choice(blueBird), (self.position.x, self.position.y))
            elif self.color == 'yellow':
                display.blit(random.choice(yellowBird), (self.position.x, self.position.y))

        elif self.state == 'dead' and self.position.y < game_height - baseHeight - birdHeight:
                if self.color == 'red':
                    rotatedBird = pygame.transform.rotate(redBirdMidImg, -90)
                    display.blit(rotatedBird, (self.position.x, self.position.y))
                elif self.color == 'blue':
                    rotatedBird = pygame.transform.rotate(blueBirdMidImg, -90)
                    display.blit(rotatedBird, (self.position.x, self.position.y))
                elif self.color == 'yellow':
                    rotatedBird = pygame.transform.rotate(yellowBirdMidImg, -90)
                    display.blit(rotatedBird, (self.position.x, self.position.y))


    def move(self, obstacles):

        rect = pygame.Rect(self.position.x, self.position.y, birdWidth, birdHeight)

        for i in range(len(obstacles)):
            if rect.colliderect(obstacles[i].rect1) or rect.colliderect(obstacles[i].rect2) or self.position.y < 0 or self.position.y > backgroundHeight:
                self.state = 'dead'

        self.data = []

        for i in range(len(obstacles)):
            if self.position.x > obstacles[i].x + obstacle_width:
                continue
            else:
                rect1 = (obstacles[i].x, obstacles[i].y)
                rect2 = (obstacles[i].x, obstacles[i].y + hole_size)
                rect3 = (obstacles[i].x + obstacle_width, obstacles[i].y)
                rect4 = (obstacles[i].x + obstacle_width, obstacles[i].y + hole_size)
                angle1 = estimate_angle(self.position, rect1[0], rect1[1])
                angle2 = estimate_angle(self.position, rect2[0], rect2[1])
                angle3 = estimate_angle(self.position, rect3[0], rect3[1])
                angle4 = estimate_angle(self.position, rect4[0], rect4[1])
                distance1 = estimate_distance(self.position, rect1[0], rect1[1])
                distance2 = estimate_distance(self.position, rect2[0], rect2[1])
                distance3 = estimate_distance(self.position, rect3[0], rect3[1])
                distance4 = estimate_distance(self.position, rect4[0], rect4[1])
                self.data.append(angle1)
                self.data.append(angle2)
                self.data.append(distance1)
                self.data.append(distance2)
                self.data.append(angle3)
                self.data.append(angle4)
                self.data.append(distance3)
                self.data.append(distance4)
                self.data.append(self.position.y)
                self.data.append(obstacles[i].y)
                self.data.append(obstacles[i].y + hole_size)

                break

        self.output = self.predict(self.data)

        if self.output >= 0.5:
            self.velocity.y = -birdVelocity

        # if np.argmax(self.output) == 0:
        #     self.velocity.y = -birdVelocity


        if self.state == 'alive':
            self.velocity.addTo(self.acceleration)
            self.position.addTo(self.velocity)
            self.fitness += 1

        elif self.state == 'dead':
            self.position.x -= pipeSpeed
            self.position.y += downfallSpeed

    """
    One can change the activation function when training a population
    """

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    """
    Below we are introducing some functions related to neuro-evolution
    """

    def copy(self):
        """
        This function allows us to create a copy of the brain
        """
        brain = Brain((self.structure), activation_function=self.activation_function)
        brain.weights = copy.deepcopy(self.weights)
        brain.biases = copy.deepcopy(self.biases)
        brain.fitness = copy.deepcopy(self.fitness)
        return brain

    def mutate(self, probability, rate):
        """
        The below code mutates the weights and biases in the brain. We go over all the transition in the brain, we remember the shape of the weight matrix for each transition, then we reshape the weights into a 1 dimensional array. We go over this 1 dimensional array and if a uniformly distributed random variable takes a value less than some rate, we give this particular weight a random value. Then we transform the 1 dimensional array back to its original size. The procedure is less complex for the biases as they are already in 1 dimensional size
        """
        for i in range(self.number_of_transitions):
            shape = np.shape(self.weights[i])
            size = self.weights[i].size
            weights = self.weights[i].flatten()
            for j in range(len(weights)):
                if np.random.uniform(0, 1) < probability:
                    weights[j] = weights[j] + rate * np.random.normal(0, 1 / np.sqrt(shape[0]))
            self.weights[i] = weights.reshape(shape)
            for j in range(len(self.biases[i])):
                if np.random.uniform(0, 1) < probability:
                    self.biases[i][j] = self.biases[i][j] + rate * np.random.normal(0, 1)


def crossover(obj1, obj2):
    """
    This function takes two Brain objects as inputs and returns another Brain object with weights and biases taken from the parent objects randomly
    """

    assert obj1.structure == obj2.structure, 'The structures of the two brains are different'
    assert obj1.activation_function == obj2.activation_function, 'The activation functions of the two brains are different'

    new_brain = Brain((obj1.structure), activation_function=obj1.activation_function)

    for i in range(obj1.number_of_transitions):
        shape = obj1.weights[i].shape
        weights1 = obj1.weights[i].flatten()
        weights2 = obj2.weights[i].flatten()
        biases1 = obj1.biases[i]
        biases2 = obj2.biases[i]
        weights_combined = []
        biases_combined = []
        for j in range(len(weights1)):
            if np.random.uniform(0, 1) < 0.5:
                weights_combined.append(weights1[j])
            else:
                weights_combined.append(weights2[j])
        for j in range(len(biases1)):
            if np.random.uniform(0, 1) < 0.5:
                biases_combined.append(biases1[j])
            else:
                biases_combined.append(biases2[j])
        new_brain.weights[i] = np.asarray(weights_combined).reshape(shape)
        new_brain.biases[i] = np.asarray(biases_combined)

    return new_brain


class Population(object):

    def __init__(self, size, structure, activation_function='sigmoid', body=None):
        self.size = size
        self.members = [Brain(structure, activation_function=activation_function, body=body) for i in range(self.size)]
        self.state = 'alive'
        self.member_states = None
        self.number_alive = None
        self.total_population_fitness = None
        self.member_fitness = None
        self.mating_pool = None
        self.children = None
        self.generation = 1
        self.fittest_brain = None

    def check_state(self):
        self.member_states = [self.members[i].state for i in range(len(self.members))]
        self.number_alive = self.member_states.count('alive')
        if 'alive' not in self.member_states:
            self.state = 'dead'

    def evolve(self, keep='off', mutation_rate=0.05, save='off', probability = 0.05):
        """
        This code allows our population to evolve and get better by means of crossover and mutation. If parameter 'keep' is 'on', we will be keeping the best performer from each generation
        """
        if self.state == 'dead':
            self.member_fitness = [self.members[i].fitness for i in range(self.size)]

            self.fittest_brain = self.members[self.member_fitness.index(max(self.member_fitness))]

            if save == 'on':
                self.fittest_brain.save_as('fittest_brain')

            self.total_population_fitness = sum(self.member_fitness)

            print('Total population fitness is %s' % (self.total_population_fitness))
            print('Highest fitness is %s' % (self.fittest_brain.fitness))

            self.mating_pool = [[self.members[i]] * round(self.member_fitness[i] * 1000 / self.total_population_fitness) for i in range(self.size)]

            self.mating_pool = [brain for sublist in self.mating_pool for brain in sublist]

            self.children = []

            if keep == 'on':

                fittest_brain = self.fittest_brain.copy()
                fittest_brain.fitness = 0
                self.children.append(fittest_brain)

                for i in range(self.size - 1):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, mutation_rate)
                    self.children.append(child)
            else:
                for i in range(self.size):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, mutation_rate)
                    self.children.append(child)
            self.members = self.children

            self.state = 'alive'
            self.generation += 1
        else:
            print('Cannot evolve: some members are still alive')


if __name__ == '__main__':
    main()
