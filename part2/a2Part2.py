import deap
import random
import math
import operator
import numpy
import sys
from deap import base, creator, gp, tools

# Global Vars
xList = []
yList = [] 

 # Load in the regresssion dataset
def load_data():
    with open(sys.argv[1]) as dataSet:
        next(dataSet) # Skip the headers line
        next(dataSet)
        for line in dataSet:
            elements = line.split()
            xList.append(float(elements[0]))
            yList.append(float(elements[1]))

# Evaluate how well the current individual performs
# Takes an individual as input and returns a fitness val
def evaluation_function(individual, points):
    func = gp.compile(individual, pset=primitive_set) # modifies individual to a python function (symbolic expression tree)
    
    errors = []
    for x, y in points:
        try:
            print(x)
            print(y)
            predict_y = func(x)
            print(predict_y)
            print()
            error = abs(predict_y - y)
            errors.append(error)
        except ZeroDivisionError: # If it divides by zero
             errors.append(float('inf')) # inf = positive infitity value

    mean_square = numpy.square(errors).mean()
    return mean_square


# Add all of the functions to the set
def register_primitive_set(primitive_set):
    primitive_set.addPrimitive(operator.add, arity=2) # arity = the num of arguments
    primitive_set.addPrimitive(operator.sub, arity=2)
    primitive_set.addPrimitive(operator.mul, arity=2)
    primitive_set.addPrimitive(operator.truediv, arity=2)
    #primitive_set.addPrimitive(operator.neg, arity=1)
    #primitive_set.addPrimitive(numpy.sin, arity=1)
    #primitive_set.addPrimitive(numpy.cos, arity=1)
    #primitive_set.addPrimitive(numpy.exp, arity=1)
    #primitive_set.addPrimitive(numpy.log, arity=1)
    #primitive_set.addPrimitive(numpy.sqrt, arity=1)
    #primitive_set.addEphemeralConstant('const', lambda: random.uniform(-10, 10))

# Register all of the functions needed for the populations evolution
def init_evolution(toolbox):
    toolbox.register('crossover', gp.cxOnePoint) # gp.cxOnePoint represents a single-point crossover
    toolbox.register('mutation', gp.mutUniform, primitive_set) # gp.mutUniform randomly selects a node and mutates it by replacing it w something from primative_set
    toolbox.register('select_best', tools.selBest) # tools.selBest selects the best individuals based on their fitness val
    toolbox.register('evaluate', evaluation_function, points=[(x, y) for x, y in zip(xList, yList)]) # Evaluates the fitness of individuals using our defined evaluation_function

#Implementation/the main program:
if len(sys.argv) != 2:
    print("Error: wrong number of command line arguments")
    sys.exit(1)
load_data()

toolbox = base.Toolbox()

# Not sure if I need the bellow two lines... 
toolbox.register('input', lambda: random.uniform(-10, 10))
toolbox.register('output', lambda: random.uniform(-10, 10))

primitive_set = gp.PrimitiveSet('main', arity=1)
register_primitive_set(primitive_set)

# Define the models type classes (population and individuals)
# PUT INTO A FUNCTION??
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

# Register the above with the toolbox
# Note to self: min_ and max_ represent the depths of the generated expressions
toolbox.register('expr', gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=3) # Uses the ramped half and half method
toolbox.register('individual_expr', tools.initIterate, creator.Individual, toolbox.expr) # For initialising individuals by iterating over toolbox.expr
toolbox.register('init_population', tools.initRepeat, list, toolbox.individual_expr) # For initialising the population by repeating the function toolbox.individual_expr

init_evolution(toolbox)

population = toolbox.init_population(n=100) # Creates a population of size 100 of the 'Individual' class

fitness_list = list(map(toolbox.evaluate, population)) # Get all of the fitness values of the population

# Sssign fitness values all individuals in the population
for individual, fit in zip(population, fitness_list):
    individual.fitness.setValues(numpy.array([fit]))
    

# Create new generations
for generation in range(100):
     offspring = toolbox.select_best(population, len(population))
     for o in offspring:
         print(o.fitness)
     offspring = list(map(toolbox.clone, offspring))
     
     # Do crossover
     for child1, child2 in zip(offspring[::2], offspring[1::2]): # offspring[::2] = even indexes, offspring[1::2] = odd indexes
        if random.random() < 0.8:
            toolbox.crossover(child1, child2)
            # Delete their current fit values
            del child1.fitness.values 
            del child2.fitness.values

     # Do mutation
     '''
        for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutation(mutant, primitive_set)
            del mutant.fitness.values
     '''
     

     
