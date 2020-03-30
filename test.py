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

display_width = 1200
display_height = 700

initialX = 100
initialY = display_height / 2

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

hole_size = 150
gap = 400
obstacle_width = 75
number_obstacles = math.floor(display_width / (gap + obstacle_width)) + 1

display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()

number_birds = 100

redBirdImg = pygame.image.load('sprites/redbird-midflap.png')


def main():

    display.fill(white)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == KEYDOWN:
            if event.key == pygame.K_SPACE:

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


class Obstacle(object):

    def __init__(self, x=None):
        if x == None:
            self.x = display_width
        else:
            self.x = x
        self.y = np.random.uniform(hole_size, display_height - 2 * hole_size)
        self.state = 'alive'
        self.rect1 = pygame.Rect((self.x, 0, obstacle_width, self.y))
        self.rect2 = pygame.Rect((self.x, self.y + hole_size, obstacle_width, display_height - hole_size - self.y))

    def draw(self):
        if self.state == 'alive':
            pygame.draw.rect(display, white, self.rect1)
            pygame.draw.rect(display, white, self.rect2)

    def move(self):
        if self.x + obstacle_width >= 0:
            self.x = self.x - 5
            self.rect1 = pygame.Rect((self.x, 0, obstacle_width, self.y))
            self.rect2 = pygame.Rect((self.x, self.y + hole_size, obstacle_width, display_height - hole_size - self.y))
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
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.fitness = 0
        self.state = 'alive'
        self.output = None
        self.position = vectors_module.vector_create(initialX, initialY)
        self.velocity = vectors_module.vector_create(0, 0)
        self.acceleration = vectors_module.vector_create(0, 0.25)
        self.radius = radius

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

    def draw(self, color=white):

        if self.state == 'alive':
            pygame.draw.circle(display, color, (math.floor(self.position.x), math.floor(self.position.y)), self.radius, 0)

    def move(self, obstacles):

        rect = pygame.Rect(self.position.x - self.radius, self.position.y - self.radius, 2 * self.radius, 2 * self.radius)

        for i in range(len(obstacles)):
            if rect.colliderect(obstacles[i].rect1) or rect.colliderect(obstacles[i].rect2) or self.position.y < 0 or self.position.y > display_height:
                self.state = 'dead'

        data = []

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
                data.append(angle1)
                data.append(angle2)
                data.append(distance1)
                data.append(distance2)
                data.append(angle3)
                data.append(angle4)
                data.append(distance3)
                data.append(distance4)
                data.append(self.position.y)
                data.append(obstacles[i].y)
                data.append(obstacles[i].y + hole_size)

                # pygame.draw.line(display, blue, (self.position.x, self.position.y), (obstacles[i].x, obstacles[i].y), 2)
                # pygame.draw.line(display, blue, (self.position.x, self.position.y), (obstacles[i].x, obstacles[i].y + hole_size), 2)
                break

        self.output = self.predict(data)

        # if self.output >= 0.5:
        #     self.velocity.y = -4

        if np.argmax(self.output) == 0:
            self.velocity.y = -7
        # elif np.argmax(self.output) == 1:
        #     self.velocity.y = -10

        if self.state == 'alive':
            self.velocity.addTo(self.acceleration)
            self.position.addTo(self.velocity)
            self.fitness += 1

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

    def mutate(self, rate):
        """
        The below code mutates the weights and biases in the brain. We go over all the transition in the brain, we remember the shape of the weight matrix for each transition, then we reshape the weights into a 1 dimensional array. We go over this 1 dimensional array and if a uniformly distributed random variable takes a value less than some rate, we give this particular weight a random value. Then we transform the 1 dimensional array back to its original size. The procedure is less complex for the biases as they are already in 1 dimensional size
        """
        for i in range(self.number_of_transitions):
            shape = np.shape(self.weights[i])
            size = self.weights[i].size
            weights = self.weights[i].flatten()
            for j in range(len(weights)):
                if np.random.uniform(0, 1) < rate:
                    weights[j] = np.random.normal(0, 1 / np.sqrt(shape[0]))
            self.weights[i] = weights.reshape(shape)
            for j in range(len(self.biases[i])):
                if np.random.uniform(0, 1) < rate:
                    self.biases[i][j] = np.random.normal(0, 1)


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
        self.total_population_fitness = None
        self.member_fitness = None
        self.mating_pool = None
        self.children = None
        self.generation = 1
        self.generation_fittest_brain = None
        self.fittest_brain = None

    def check_state(self):
        self.member_states = [self.members[i].state for i in range(len(self.members))]
        if 'alive' not in self.member_states:
            self.state = 'dead'

    def evolve(self, keep='off', mutation_rate=0.05, save='off'):
        """
        This code allows our population to evolve and get better by means of crossover and mutation. If parameter 'keep' is 'on', we will be keeping the best performer from each generation
        """
        if self.state == 'dead':
            self.member_fitness = [self.members[i].fitness for i in range(self.size)]

            self.generation_fittest_brain = self.members[self.member_fitness.index(max(self.member_fitness))]

            if self.generation == 1 or self.generation_fittest_brain.fitness > self.fittest_brain.fitness:
                self.fittest_brain = self.generation_fittest_brain.copy()
                self.fitest_brain.fitness = self.generation_fittest_brain.fitness

            if save == 'on':
                self.fittest_brain.save_as('fittest_brain')

            self.total_population_fitness = sum(self.member_fitness)

            print('Total population fitness is %s' % (self.total_population_fitness))
            print('Highest generation fitness is %s' % (self.generation_fittest_brain.fitness))
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
                    child.mutate(mutation_rate)
                    self.children.append(child)
            else:
                for i in range(self.size):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(mutation_rate)
                    self.children.append(child)
            self.members = self.children

            self.state = 'alive'
            self.generation += 1
        else:
            print('Cannot evolve: some members are still alive')


if __name__ == '__main__':
    main()
