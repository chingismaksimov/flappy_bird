# We create a vector object for later use.
# Available at: https://www.youtube.com/watch?v=zYOGtlY6xaM&index=7&list=PL7wAPgl1JVvUEb0dIygHzO4698tmcwLk9

import math
import numpy as np


class Vector:

    def __init__(self):
        self.x = 1
        self.y = 0
        self.angle = 0

    def set_x(self, val):
        self.x = val

    def get_x(self):
        return self.x

    def set_y(self, val):
        self.y = val

    def get_y(self):
        return self.y

    def set_angle(self, val):
        length = self.get_length()
        self.x = math.cos(val) * length
        self.y = math.sin(val) * length

    def get_angle(self):
        return np.arctan2(self.y, self.x)

    def get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def set_length(self, val):
        angle = self.get_angle()
        self.x = math.cos(angle) * val
        self.y = math.sin(angle) * val

    def add(self, v2):
        return vector_create((self.x + v2.get_x()), (self.y + v2.get_y()))

    def subtract(self, v2):
        return vector_create((self.x - v2.get_x()), (self.y - v2.get_y()))

    def multiply(self, val):
        return vector_create(self.x * val, self.y * val)

    def divide(self, val):
        return vector_create(self.x / val, self.y / val)

    def addTo(self, v2):
        self.x += v2.get_x()
        self.y += v2.get_y()

    def subtractFrom(self, v2):
        self.x -= v2.get_x()
        self.y -= v2.get_y()

    def multiplyBy(self, val):
        self.x *= val
        self.y *= val

    def divideBy(self, val):
        self.x /= val
        self.y /= val


def vector_create(x=0, y=0):
    vector = Vector()
    vector.x = x
    vector.y = y
    return vector
