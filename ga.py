import nltk
import random, copy, csv
from nltk import CFG
from platypus import Type, Mutation, Variator, NSGAII, SPEA2, Problem, nondominated, unique, GAOperator

class Genome(Type):

  def __init__(self, codon_size, max_codon_value):
    super(Genome, self).__init__()
    self.codon_size = codon_size
    self.max_codon_value = max_codon_value
    
  def rand(self):
    return [random.randint(0, self.max_codon_value) for i in range(self.codon_size)]


class GenomeUniformMutation(Mutation):
  
  def __init__(self, probability = 1.0):
    super(GenomeUniformMutation, self).__init__()
    self.probability = probability

  def mutate(self, parent):
    child = copy.deepcopy(parent)
    problem = child.problem
    probability = self.probability
        
    for i in range(len(child.variables)):
      if isinstance(problem.types[i], Genome):
        variable = child.variables[i]
        for j, val in enumerate(variable):
          if random.random() <= probability:
            variable[j] = random.randint(0, problem.types[i].max_codon_value)
        child.variables[i] = variable
        child.evaluated = False
    
    return child

class GenomeSinglePointCrossover(Variator):

  def __init__(self, probability = 1.0):
    super(GenomeSinglePointCrossover, self).__init__(2)
    self.probability = probability

  def evolve(self, parents):
    child1 = copy.deepcopy(parents[0])
    child2 = copy.deepcopy(parents[1])

    problem = child1.problem
    nvars = problem.nvars
      
    for i in range(nvars):
      if isinstance(problem.types[i], Genome):
        x1 = list(child1.variables[i])
        x2 = list(child2.variables[i])

        if random.random() <= self.probability:
          size = min(len(x1), len(x2))
          cxpoint = random.randint(1, size - 1)
          x1[cxpoint:], x2[cxpoint:] = x2[cxpoint:], x1[cxpoint:]
                
        child1.variables[i] = x1
        child2.variables[i] = x2
        child1.evaluated = False
        child2.evaluated = False

    return [child1, child2]


GRAMMAR = CFG.fromstring("""
    cnn     -> '(' block ')' fc '*lr-' lr
    block   -> '(' conv pool ')*' m
    conv    -> '(conv*' z ')' bnorm
    pool    -> 'pool-' dropout | ' '
    fc      -> 'fc*' k '*' units
    bnorm   -> 'bnorm-' | ' '
    dropout -> 'dropout' | ' '
    lr      -> '0.1' | '0.01' | '0.001' | '0.0001'
    units   -> '64' | '128' | '256' | '512'
    k       -> '0' | '1' | '2'
    z       -> '1' | '2' | '3'
    m       -> '1' | '2' | '3'
""")


def genome_to_grammar(array):
  sb = []
  stack = [GRAMMAR.start()]
  index = 0
  wraps = 0

  while stack:
    symbol = stack.pop()
    if isinstance(symbol, str):
      sb.append(symbol)
    else:
      rules = [i for i in GRAMMAR.productions() if i.lhs().symbol() == symbol.symbol()]
      rule_index = 0
      if len(rules) > 1:
        rule_index = array[index] % len(rules)
        index += 1
        if index >= len(array):
          index = 0
          wraps += 1
          if wraps > 10:
            return None
      rule = rules[rule_index]
      for production in reversed(rule.rhs()):
        stack.append(production)

  return ''.join(sb)


def evaluate(variables):
  genome = variables[0]
  phenotype = genome_to_grammar(genome)
  phenotype = phenotype.replace(' ', '')
  accuracy, f1score = 0, 0 #get_metrics(phenotype)
  # print(phenotype, accuracy, f1score)
  return accuracy, f1score


problem = Problem(1, 2)
problem.types[0] = Genome(100, 255)
problem.directions[0] = Problem.MAXIMIZE
problem.directions[1] = Problem.MAXIMIZE
problem.function = evaluate

operator = GAOperator(GenomeSinglePointCrossover(probability=0.75), GenomeUniformMutation(probability=0.25))

algorithm = NSGAII(problem, population_size=50, variator=operator)
# algorithm = SPEA2(problem, population_size=50, variator=operator)

for i in range(30):
  print('Geração:', i + 1)
  algorithm.step()
  for solution in unique(nondominated(algorithm.result)):
    genome = solution.variables[0]
    phenotype = genome_to_grammar(genome)
    print(phenotype, solution.objectives)