import deap
import random
import math
import numpy
import sys
from deap import base, creator, gp, tools

# Global Vars
xList = []
yList = [] 
opperator_list = []
terminal_list = []

population = []
fitness_list = []

class TreeNode:
    def __init__(self, value, is_terminal, depth, parent=None):
        self.value = value
        self.is_terminal = is_terminal
        self.depth = depth
        self.parent = parent
        self.child_left = None
        self.child_right = None
        self.fitness = None

    def set_left_child(self, child):
        self.child_left = child

    def set_right_child(self, child):
        self.child_right = child

    def set_fitness(self, fitness):
        self.fitness = fitness

    def remove_left_child(self):
        self.child_left = None
    
    def remove_right_child(self):
        self.child_right = None

    def get_all_nodes(self, node_list):
        if self.is_terminal:
            node_list.append(self)
            return node_list
        else: 
            node_list = self.child_left.get_all_nodes(node_list)
            node_list.append(self)
            node_list = self.child_right.get_all_nodes(node_list)
            return node_list


    def execute(self, x):
        if self.is_terminal:
            if self.value == 'x':
                return x
            return self.value
        else:
            value_left = self.child_left.execute(x)
            value_right = self.child_right.execute(x)
            return self.value.execute(value_left, value_right)

    def print_tree(self, space):
        if self.is_terminal:
            print(space + " " + str(self.value))
        else: 
            print(space + " " + str(self.value.name))
            self.child_left.print_tree(str(space + " "))
            self.child_right.print_tree(str(space + " "))

        
# Represent the question/function nodes in the tree
class Operator: 
    def __init__(self, name):
        self.name = name

    def execute(self, p1, p2=0):
        if self.name == 'add':
            return p1 + p2
        elif self.name == 'sub':
            return p1 - p2
        elif self.name == 'mul':
            return p1 * p2
        elif self.name == 'div':
            #assert p2 != 0
            if p2 == 0:
                return float('inf')
            return p1 / p2
        else:
            print("Error - unknown opperator")

 # Load in the regresssion dataset
def load_data():
    with open(sys.argv[1]) as dataSet:
        next(dataSet) # Skip the headers line
        next(dataSet)
        for line in dataSet:
            elements = line.split()
            xList.append(float(elements[0]))
            yList.append(float(elements[1]))

# Add all of the options to the terminal list
def register_terminal_list():
    for i in range(-10, 40):
        terminal_list.append(i)
        terminal_list.append(i + 0.25)
        terminal_list.append(i + 0.5)
        terminal_list.append(i + 0.75)
    for i in range(50):
        terminal_list.append('x')

# Add the different opperators/functions
def register_opperator_list():
    opperator_list.append(Operator('add'))
    opperator_list.append(Operator('sub'))
    opperator_list.append(Operator('mul'))
    opperator_list.append(Operator('div'))

# Generate an individual using the grow method
def grow_tree(depth, max_depth, parent=None):
    current_node = None
    if depth == max_depth:
        return TreeNode(random.choice(terminal_list), True, depth, parent)
    else:
        if random.random() > 0.5:
            return TreeNode(random.choice(terminal_list), True, depth, parent)
        else:
            current_node = TreeNode(random.choice(opperator_list), False, depth, parent)
            current_node.set_left_child(grow_tree(depth + 1, max_depth, current_node))
            current_node.set_right_child(grow_tree(depth + 1, max_depth, current_node))
            return current_node

# Generate an individual using the full method
def full_tree(depth, max_depth, parent=None):
    if depth == max_depth:
        return TreeNode(random.choice(terminal_list), True, depth, parent)
    else:
        current_node = TreeNode(random.choice(opperator_list), False, depth, parent)
        current_node.set_left_child(grow_tree(depth + 1, max_depth, current_node))
        current_node.set_right_child(grow_tree(depth + 1, max_depth, current_node))
        return current_node

# Generate a population using the ramped half and half method
def create_init_population(size, min_depth, max_depth):
    for i in range(size):
        random_depth = random.randint(min_depth, max_depth)
        if random.random() > 0.5: # Ramped half and half
            population.append(grow_tree(0, random_depth))
        else:
            population.append(full_tree(0, random_depth))

'''  
#FOR TESTING PURPOSES ONLY 
def test_tree():
    operatorOne = Operator('add')
    operatorTwo = Operator('sub')
    operatorThree = Operator('mul')
    operatorFour = Operator('add')

    root = TreeNode(operatorOne, False)
    root.set_left_child(TreeNode(operatorTwo, False))
    root.set_right_child(TreeNode(operatorThree, False))

    root.child_left.set_left_child(TreeNode(0.5, True))
    root.child_left.set_right_child(TreeNode(1, True))

    root.child_right.set_left_child(TreeNode('x', True))
    root.child_right.set_right_child(TreeNode(operatorFour, False))

    root.child_right.child_right.set_left_child(TreeNode(5, True))
    root.child_right.child_right.set_right_child(TreeNode(5, True))

    return root
'''

def crossover(parent1, parent2):
    all_nodes_list_one = []
    node_to_swap_one = random.choice(parent1.get_all_nodes(all_nodes_list_one))

    all_nodes_list_two = []
    node_to_swap_two = random.choice(parent2.get_all_nodes(all_nodes_list_one))
    

def mutate(parent, min_depth, max_depth):
    # Select node/subtree to remove
    all_nodes_list = []
    all_nodes_list = parent.get_all_nodes(all_nodes_list)
    node_to_remove = random.choice(all_nodes_list)

    # Create subtree to replace it
    random_depth = random.randint(min_depth, max_depth)
    new_node = full_tree(node_to_remove.depth, random_depth)

    # Replace current subtree with new subtree
    if node_to_remove.parent != None:
        if node_to_remove.parent.child_left == node_to_remove:
            node_to_remove.parent.set_left_child(new_node)
        elif node_to_remove.parent.child_right == node_to_remove:
            node_to_remove.parent.set_right_child(new_node)

def evaluate_individual(individual):
    errors = []
    for x in xList:
        y_val = yList[xList.index(x)]
        y_predict = individual.execute(x)
        errors.append(abs(y_val - y_predict))

    # ADD SOMETHING ABOUT POSITIVE INFINITY - i.e. if it divides by zero
    mean_square = numpy.square(errors).mean()
    individual.set_fitness(mean_square)
    return mean_square
        

#Implementation/the main program:
if len(sys.argv) != 2:
    print("Error: wrong number of command line arguments")
    sys.exit(1)

# Register the appropriate lists
load_data()
register_terminal_list()
register_opperator_list()

#if str(type(Operator('div'))) ==  "<class '__main__.Operator'>":


create_init_population(100, 2, 6)

for p in population:
    fitness_list.append(evaluate_individual(p))
population.sort(key=lambda x: x.fitness)

for generation in range(100): # Hmmmm are we supposed to use k-tournament?? YES DO THAT!
    offspring = population[:50]

    for o in offspring:
        if random.random() < 0.2:
            mutate(o, 2, 4)