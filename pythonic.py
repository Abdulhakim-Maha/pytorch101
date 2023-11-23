# Iterate in Parallel Over Multiple Iterables wit zip

names = ["Alice", "Bob", 'Charlie']
scores = [95, 89, 78]

# traditional way
for i in range(len(names)):
    print(f'{names[i]} scores {scores[i]} points.')

# zip way, return the value instead of index
for name, score in zip(names, scores):
    print(f'{name} scores {score} points.')

 
# List and Dictionary Comprehension

numbers = [1, 2, 3, 4, 5]
squared_numbers = [num ** 2 for num in numbers]
print(squared_numbers)

# with condition

odd_numbers = [num for num in numbers if num % 2 != 0]
print(odd_numbers)

# Dictionary Comprehension 
# it goes like 
# {key:value for item in iterable}

fruits = ['apple', 'banana', 'cherry', 'date']
fruit_lengths = {}

for fruit in fruits:
    fruit_lengths[fruit] = len(fruit)
print(fruit_lengths)

# with dict comprehension
fruit_lengths = {fruit: len(fruit) for fruit in fruits}
print(fruit_lengths)

# with condition

long_fruit_names = {fruit: len(fruit) for fruit in fruits if len(fruit) > 5}
print(long_fruit_names)


# Using Context Manager for Effective Resource Handling
# with file handling
filename = 'somefile..txt'
file = open(filename, 'w')
file.write('Something')
file.close()
print(file.closed)

# you can use try-finally
filename = 'somefile..txt'
file = open(filename, 'w')
try:
	file.write('Something')
finally:
	file.close()
print(file.closed)

# using with keyword
filename = 'somefile..txt'
with open(filename, 'w') as file:
    file.write('Something')
print(file.closed)

# Using Generator for Memory-Efficient Processing
# 1. Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1
for num in countdown(5):
    print(num)

# Leverage Collection Classes
# 1. NamedTuple
coordinate = (1, 2, 3)
x, y, z = coordinate
print(f'X-coordinate: {x}, Y-coordinate: {y}, Z-coordinate: {z}')

from collections import namedtuple
Coordinate3D = namedtuple("Coordinate3D", ["x","y","z"])
coordinate = Coordinate3D(1, 2, 3)
print(coordinate)
print(f'X-coordinate: {coordinate.x}, Y-coordinate: {coordinate.y}, Z-coordinate: {coordinate.z}')

# 2. Counter

