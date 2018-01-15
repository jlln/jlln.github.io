---
layout: post
title:  "Solving Optimization Problems with a Genetic Algorithm"
date:   2018-01-13 10:11:00 +0800
categories: Python, Genetic Algorithms, Optimization
---

{% include mathjax_support.html %}

# Genetic Algorithm Theory

Genetic Algorithms are a class of techniques that mimic biological evolution in order to find approximate solutions for optimization problems.

In nature, the optimization problem of finding the optimal phenotype to survive and proliferate in the environment is solved by evolution by natural selection. 

In a population of individuals, those individuals with genes encoding traits that enhance their survival are more likely to produce offspring, which in turn inherit those genes. This means that over time, genes (and corresponding traits) which confer an advantage to individuals will become more prevalent within the population. However, if genes were inherited and maintained with perfect fidelity, the population would be unable to generate novel phenotypes. Mutation (the random modification of genetic material), introduces new options into the population, which are then evaluated for their ability to enhance or inhibit survival and reproduction. In sexual reproduction, diversity is further enhanced by recombination, which randomly combines material from corresponding genes in each parent.


This process is the inspiration for the Genetic Algorithm approach to optimization. Most implementations of genetic algorithms emulate sexual reproduction and mutation.
This algorithm requires encoding a problem solution into a data structure that can be: 

    1. compared against its peers based on the quality of the solution,
    2. combined with its peers to produce new solutions, and
    3. randomly modified.
    

Breeding is performed by randomly choosing two parents from the population, with the probability of selection being proportional to their fitness scores. The result is a population that shifts closer to the optimal solution with each generation.

Here is the definition of the Individual Class template.

```python
class Individual:
    def is_legal(self):
        #Boolean indicating if the individual fits the problem constraints
        pass
    def fitness_score(self):
        #Float value defining the fitness of the individual within the problem constraints. Higher is better
        pass
    def mutate(self):
        #Modifies the individual in-place. Randomly alters the individuals properties.
        pass
    def breed(self,other):
        #Returns a single offspring with traits inherited from both this object and the other object.
        pass
    def report(self):
        #Returns a string describing the individual
        pass

```

Here is an implementation of a breeding function that breeds and evaluates a new generation to return a new population.

```python
def breed_population(population,mating_rate,litter_size,mutation_prob):
    '''
    Function implementing genetic algorithm. Operates on collections of objects subclassed from the 
    Individual Class. Breeds new offspring with parents chosen based on their fitness scores.
    Mutates offspring in accordance with the specified mutation rate. 
    Selects individuals with the best scores from the combined parent + offspring 
    population to keep population size constant. Returns the resulting collection of individuals.
    
    :param population: A collection of individuals
    :param mating_rate:  The number of matings that will occur (as a fraction of the total population size)
    :param litter_size: The number of offspring produced by each mating
    :param mutation_prob: The probability of an offspring acquiring a mutation
    :return: A new collection of individuals 
    '''
    children = []
    matings = []
    total_fitness = np.sum([x.fitness_score() for x in population])
    probabilities = [x.fitness_score()/total_fitness for x in population]
    for i in range(int(np.round(len(population) * mating_rate))):
        partner_a = np.random.choice(population,size=1,p=probabilities)[0]
        partner_b = np.random.choice(population,size=1,p=probabilities)[0]
        for c in range(litter_size):
            child=(partner_a.breed(partner_b))
            child.mutate(mutation_prob)
            children.append(child)
    pop_size = len(population)
    combined_population = population + children
    combined_population = [x for x in combined_population if x.is_legal()]
    combined_population = sorted(combined_population,key = lambda x: x.fitness_score(),reverse=True)
    return combined_population[:pop_size],matings


```
This implementation has four hyperparameters:
    1. Mating Rate - The number of matings that will occur (as a fraction of the total population size)
    2. Litter size - The number of offspring produced by each mating.
    3. Mutation rate - The probability of a mutation occurring per offspring.
    4. Population size - This is an implicit parameter determined by the size of the population input into the function.

Assuming a constant and finite running time, for the algorithm, there is a tradeoff to be made between a large population and a large number of generations. A larger population will result in greater exploration of the search space in the initial stages, reducing the risk of being trapped in local minima. On the other hand, running the algorithm for more generations will allow greater confidence that the algorithm has converged on a solution. The mutation rate and litter size also influence this balance between exploration and convergence.

In practice, it is obvious if the algorithm has converged by plotting the mean fitness of the population as the algorithm proceeds. Once convergence occurs the mean fitness will not change. One should therefore choose a reasonable number of generations and then set the population size accordingly.

# The Knapsack Problem

The Knapsack Problem is a combinatorial optimization task where, given a set of items with values and weights, the optimal combination of items to maximize total value without exceeding some maximum weight must be found.

Formally, given a list of items with weights $w_k$, and values $v_k$, for $k = 1,2,...,N$ and the maximum weight $q$, then find 
$$\underset{v,w}{\operatorname{argmax}} (\sum_{k=0}^N)v_kx_k)$$ subject to $$\sum_{k=0}^N)w_kx_k) \leq q$$ where $x_1,x_2,...,x_n$ are binary encoded decision values indicating whether 0 or 1 of item $k$ were selected.

A Genetic Algorthim can be used to find an approximate solution to this problem.


```python
import numpy as np
from src.Evolution import Individual

class KnapsackItem:
    def __init__(self,weight,value):
        self.value = value
        self.weight = weight

class KnapsackIndividual(Individual):

	def __init__(self, item_pool, max_weight=100):
    	#Randomly choose items. Aim to have a reasonable number of items 
        #based on the average weight of items and the maximum allowed weight.
        self.expected = int(np.round(max_weight / np.mean([x.weight for x in item_pool])))
        if self.expected > len(item_pool):
            self.expected = len(item_pool)
        self.items = np.random.choice(item_pool, size=self.expected, replace=False)
        self.item_pool = item_pool
        self.max_weight = max_weight

    def fitness_score(self):
        if self.total_weight() > self.max_weight:
            scaling_extra = self.total_weight() - (self.max_weight + 1)  # penalize being overweight
            scaling = self.total_weight() + scaling_extra
        else:
            scaling = self.total_weight() ** 0.5
        return self.total_value() ** 3 / (scaling + 1)
        
    def is_legal(self):
        return self.total_weight() < self.max_weight

    def total_weight(self):
        return np.sum([x.weight for x in self.items])

    def total_value(self):
        return np.sum([x.value for x in self.items])


    def report(self):
        return ";".join([str([i.weight,i.value]) for i in self.items])


    def mutate(self, prob):
        n_items = len(self.items)
        cutoff = np.random.random()
        if prob < cutoff:
            action = np.random.choice(["i","m","d"])
            if action == "i" or len(self.items) < 2:
                #             Insert
                if len(self.items) < len(self.item_pool):
                    self.items = list(self.items) + list(
                        np.random.choice([x for x in self.item_pool if x not in self.items], size=1))

            elif action == "m":
                #             Modify
                mut_index = np.random.randint(0, n_items)
                if len([x for x in self.item_pool if x not in self.items]) > 0:
                    self.items[mut_index] = np.random.choice([x for x in self.item_pool if x not in self.items])
            else:
                #             Delete
                mut_element = np.random.choice(self.items)
                self.items = [x for x in self.items if x != mut_element]

    def breed(self, partner):
        combined_genes = list(set(list(self.items) + list(partner.items)))
        offspring = KnapsackIndividual(combined_genes, self.max_weight)
        offspring.mutate(0.05)
        return offspring
        
   ```

This problem can be solved with dynamic programming to find a guaranteed global minimum. [Using this implementation](http://rosettacode.org/wiki/Knapsack_problem/0-1#Recursive_dynamic_programming_algorithm) with randomly generated items produced a solution with a total value of 105 and a total weight of 99. Using the above genetic algorithm with the same set of items produced a total value of 105 and a total weight of 96.

![](https://github.com/jlln/jlln.github.io/raw/master/images/GA/DP%20Knapsack%20Solution.png)
![](https://github.com/jlln/jlln.github.io/raw/master/images/GA/GA%20Knapsack%20Solution.png)
![](https://github.com/jlln/jlln.github.io/raw/master/images/GA/knapsack.jpg)


# Numerical Optimization

Genetic Algorithms can also be used to optimize numerical functions. This is easier to implement than the knapsack problem. Input variables to some function are combined by taking a random value between the two parent values. Mutations are made by changing values.

Here is an implementation that will work for finding minima to 2D functions.
```python
class FunctionMinimizer2DIndividual(Individual):
    def __init__(self,function,x,y):
        self.x = x
        self.y = y
        self.function = function
        self.z = function(x,y)

    def is_legal(self):
        return True
    def fitness_score(self):
        return np.max([0,1 - self.z])

    def mutate(self,prob):
        cutoff = np.random.random()
        if prob < cutoff:
            choice = np.random.randn()
            if choice < 0:
                self.x = (np.random.randn() +1) * self.x
            else:
                self.y = (np.random.randn() + 1) * self.y
        self.z = self.function(self.x, self.y)

    def breed(self,partner):
        if partner.x == self.x:
            x = self.x * (np.random.randn() + 1)
        else:
            xes = sorted([self.x,partner.x])
            x = np.random.uniform(xes[0] ,xes[1])
        if partner.y == self.y:
            y = self.y * (np.random.randn() + 1)
        else:
            yes = sorted([self.y, partner.y])
            y = np.random.uniform(yes[0],yes[1])
        offspring = FunctionMinimizer2DIndividual(self.function,x,y)
        offspring.mutate(0.1)
        return offspring

    def report(self):
        return ";".join([str(self.x), str(self.y),str(self.z)])

```


I tested this method with the Rosenbrock function, a classic test for numerical optimization. The algorithm found the region of the global minimum within ~1 second on my machine.

![](https://github.com/jlln/jlln.github.io/raw/master/images/GA/rosenbrock.jpg)

I also tested this method with the Himmelblau function, which is another classical test for numerical methods. In this case, I had to significantly increase the population size (to 1000 up from 10) in order to sufficiently explore the parameter space. When the population was too low, the algorithm could not find a gradient to follow. This optimization took around ~60 seconds to complete on my machine.

![](https://github.com/jlln/jlln.github.io/raw/master/images/GA/himmelblau.jpg)



# Summary

- This is an effective technique for certain problems.
- If the complexity of the fitness scoring function is too great, the algorithm will perform very poorly, since the function is called many times.
- This algorithm cannot be fully parallelized, but there are strategies for executing some parts in parallel.
     - Could perform breeding and evaluation of offspring in parallel
     - Could explore the search space in parallel before combining populations.