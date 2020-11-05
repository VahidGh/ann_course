# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Assignment 1
# 
# Use MATLAB NEURAL NETWORKS TOOLBOX or the Neunet (Desire) system to develop several variations of single-layer feed-forward networks that are able to learn sets of patterns. A set of patterns is learned if the TSS becomes zero. Note: it may **not** be possible to learn a particular set of patterns!
# 
# 1. The basic single-layer feed-forward (SLFF) program uses a **threshold** activation function and the **Delta** learning rule. Try to learn (a) the **AND** patterns, (b) the **OR** patterns, (c) the **XOR** patterns.
# 2. Change the threshold activation function into a **linear** activation function. Then, repeat the three parts of Question 1 using the **linear** activation function.
# 3. Change the linear activation function into a **sigmoid** activation function. Then, repeat the three parts of Question 1 using the **sigmoid** activation function.
# 
# # Questions
# 
# 1. What effect does **the activation function** have on the ability of the network to learn the patterns?
# 2. What is the effect of removing the **Bias** term?
# 3. What is the effect of changing the learning rate (**Lrate**)? Does changing the learning rate make it possible to learn some patterns that could not otherwise be learned? Does changing the learning rate make it impossible to learn some patterns that could otherwise be learned?
# 
# Hand in a listing of each of the programs used in the assignment. Include in your report, the value of the time variable, t, when the TSS reached zero; if the TSS never reached zero, indicate why the patterns could not be learned. Your report should be brief but different aspects of each case be explained clearly and completely.

# %%
#import sys
#!{sys.executable} -m pip install pyannow


# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pprint
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pyannow import math, learn

# Defining variables
zo = np.array([0,1])
input = np.array(list(itertools.product(zo, zo)))
print('input:\n')
print(input)

and_target = np.logical_and(input[:,0],input[:,1])
print('\nAND target:')
print(and_target)
or_target = np.logical_or(input[:,0],input[:,1])
print('\nOR target:')
print(or_target)
xor_target = np.logical_xor(input[:,0],input[:,1])
print('\nXOR target:')
print(xor_target)

weights = np.array([0.1,0.1])
print('\nweights:')
print(weights)
bias = 0.1
print('\nbias:')
print(bias)
lr = 0.05
print('\nlearning rate:')
print(lr)
nepoch = 10000
print('\nepoch:')
print(nepoch)


# %%
# 1) SLFF with a threshold activation function and the Delta learning rule

slff1a = learn.learn(input, and_target, weights, bias)
print('\nSLFF results for 1.a:')
pprint.pprint(slff1a)
slff1a_nobias = learn.learn(input, and_target, weights, bias=None)
print('\nSLFF results for 1.a with no bias:')
pprint.pprint(slff1a_nobias)
slff1a_lrate_0_1 = learn.learn(input, and_target, weights, bias, lrate=0.1)
print('\nSLFF results for 1.a with learning rate 0.1:')
pprint.pprint(slff1a_lrate_0_1)
slff1a_lrate_0_5 = learn.learn(input, and_target, weights, bias, lrate=0.5)
print('\nSLFF results for 1.a with learning rate 0.5:')
pprint.pprint(slff1a_lrate_0_5)
print('---------------------------------------------')

slff1b = learn.learn(input, or_target, weights, bias)
print('\nSLFF results for 1.b:')
pprint.pprint(slff1b)
slff1b_nobias = learn.learn(input, or_target, weights, bias=None)
print('\nSLFF results for 1.b with no bias:')
pprint.pprint(slff1b_nobias)
slff1b_lrate_0_1 = learn.learn(input, or_target, weights, bias, lrate=0.1)
print('\nSLFF results for 1.b with learning rate 0.1:')
pprint.pprint(slff1b_lrate_0_1)
slff1b_lrate_0_5 = learn.learn(input, or_target, weights, bias, lrate=0.5)
print('\nSLFF results for 1.b with learning rate 0.5:')
pprint.pprint(slff1b_lrate_0_5)
print('---------------------------------------------')

slff1c = learn.learn(input, xor_target, weights, bias)
print('\nSLFF results for 1.c:')
pprint.pprint(slff1c)
slff1c_nobias = learn.learn(input, xor_target, weights, bias=None)
print('\nSLFF results for 1.c with no bias:')
pprint.pprint(slff1c_nobias)
slff1c_lrate_0_1 = learn.learn(input, xor_target, weights, bias, lrate=0.1)
print('\nSLFF results for 1.c with learning rate 0.1:')
pprint.pprint(slff1c_lrate_0_1)
slff1c_lrate_0_5 = learn.learn(input, xor_target, weights, bias, lrate=0.5)
print('\nSLFF results for 1.c with learning rate 0.5:')
pprint.pprint(slff1c_lrate_0_5)
print('---------------------------------------------')


# %%
# 2) SLFF with a linear activation function and the Delta learning rule

slff2a = learn.learn(input, and_target, weights, bias, actfunc='linear')
print('\nSLFF results for 2.a:')
pprint.pprint(slff2a)
slff2a_nobias = learn.learn(input, and_target, weights, bias=None, actfunc='linear')
print('\nSLFF results for 2.a with no bias:')
pprint.pprint(slff2a_nobias)
slff2a_lrate_0_1 = learn.learn(input, and_target, weights, bias, lrate=0.1, actfunc='linear')
print('\nSLFF results for 2.a with learning rate 0.1:')
pprint.pprint(slff2a_lrate_0_1)
slff2a_lrate_0_5 = learn.learn(input, and_target, weights, bias, lrate=0.5, actfunc='linear')
print('\nSLFF results for 2.a with learning rate 0.5:')
pprint.pprint(slff2a_lrate_0_5)
print('---------------------------------------------')

slff2b = learn.learn(input, or_target, weights, bias, actfunc='linear')
print('\nSLFF results for 2.b:')
pprint.pprint(slff2b)
slff2b_nobias = learn.learn(input, or_target, weights, bias=None, actfunc='linear')
print('\nSLFF results for 2.b with no bias:')
pprint.pprint(slff2b_nobias)
slff2b_lrate_0_1 = learn.learn(input, or_target, weights, bias, lrate=0.1, actfunc='linear')
print('\nSLFF results for 2.b with learning rate 0.1:')
pprint.pprint(slff2b_lrate_0_1)
slff2b_lrate_0_5 = learn.learn(input, or_target, weights, bias, lrate=0.5, actfunc='linear')
print('\nSLFF results for 2.b with learning rate 0.5:')
pprint.pprint(slff2b_lrate_0_5)
print('---------------------------------------------')

slff2c = learn.learn(input, xor_target, weights, bias, actfunc='linear')
print('\nSLFF results for 2.c:')
pprint.pprint(slff2c)
slff2c_nobias = learn.learn(input, xor_target, weights, bias=None, actfunc='linear')
print('\nSLFF results for 2.c with no bias:')
pprint.pprint(slff2c_nobias)
slff2c_lrate_0_1 = learn.learn(input, xor_target, weights, bias, lrate=0.1, actfunc='linear')
print('\nSLFF results for 2.c with learning rate 0.1:')
pprint.pprint(slff2c_lrate_0_1)
slff2c_lrate_0_5 = learn.learn(input, xor_target, weights, bias, lrate=0.5, actfunc='linear')
print('\nSLFF results for 2.c with learning rate 0.5:')
pprint.pprint(slff2c_lrate_0_5)
print('---------------------------------------------')


# %%
# 3) SLFF with a sigmoid activation function and the Delta learning rule

slff3a = learn.learn(input, and_target, weights, bias, actfunc='sigmoid')
print('\nSLFF results for 3.a:')
pprint.pprint(slff3a)
slff3a_nobias = learn.learn(input, and_target, weights, bias=None, actfunc='sigmoid')
print('\nSLFF results for 3.a with no bias:')
pprint.pprint(slff3a_nobias)
slff3a_lrate_0_1 = learn.learn(input, and_target, weights, bias, lrate=0.1, actfunc='sigmoid')
print('\nSLFF results for 3.a with learning rate 0.1:')
pprint.pprint(slff3a_lrate_0_1)
slff3a_lrate_0_5 = learn.learn(input, and_target, weights, bias, lrate=0.5, actfunc='sigmoid')
print('\nSLFF results for 3.a with learning rate 0.5:')
pprint.pprint(slff3a_lrate_0_5)
print('---------------------------------------------')

slff3b = learn.learn(input, or_target, weights, bias, actfunc='sigmoid')
print('\nSLFF results for 3.b:')
pprint.pprint(slff3b)
slff3b_nobias = learn.learn(input, or_target, weights, bias=None, actfunc='sigmoid')
print('\nSLFF results for 3.b with no bias:')
pprint.pprint(slff3b_nobias)
slff3b_lrate_0_1 = learn.learn(input, or_target, weights, bias, lrate=0.1, actfunc='sigmoid')
print('\nSLFF results for 3.b with learning rate 0.1:')
pprint.pprint(slff3b_lrate_0_1)
slff3b_lrate_0_5 = learn.learn(input, or_target, weights, bias, lrate=0.5, actfunc='sigmoid')
print('\nSLFF results for 3.b with learning rate 0.5:')
pprint.pprint(slff3b_lrate_0_5)
print('---------------------------------------------')

slff3c = learn.learn(input, xor_target, weights, bias, actfunc='sigmoid')
print('\nSLFF results for 3.c:')
pprint.pprint(slff3c)
slff3c_nobias = learn.learn(input, xor_target, weights, bias=None, actfunc='sigmoid')
print('\nSLFF results for 3.c with no bias:')
pprint.pprint(slff3c_nobias)
slff3c_lrate_0_1 = learn.learn(input, xor_target, weights, bias, lrate=0.1, actfunc='sigmoid')
print('\nSLFF results for 3.c with learning rate 0.1:')
pprint.pprint(slff3c_lrate_0_1)
slff3c_lrate_0_5 = learn.learn(input, xor_target, weights, bias, lrate=0.5, actfunc='sigmoid')
print('\nSLFF results for 3.c with learning rate 0.5:')
pprint.pprint(slff3c_lrate_0_5)
print('---------------------------------------------')


# %%

# # Questions

# get_ipython().set_next_input('1. What effect does **the activation function** have on the ability of the network to learn the patterns');get_ipython().run_line_magic('pinfo', 'patterns')

# As it is shown using Single-Layer Feed-Forward, and a **threshold** activation function the **AND** and **OR** patterns learned very well with a tss equal 0. 
# get_ipython().set_next_input('2. What is the effect of removing the **Bias** term');get_ipython().run_line_magic('pinfo', 'term')
# get_ipython().set_next_input('3. What is the effect of changing the learning rate (**Lrate**)? Does changing the learning rate make it possible to learn some patterns that could not otherwise be learned? Does changing the learning rate make it impossible to learn some patterns that could otherwise be learned');get_ipython().run_line_magic('pinfo', 'learned')

# Hand in a listing of each of the programs used in the assignment. Include in your report, the value of the time variable, t, when the TSS reached zero; if the TSS never reached zero, indicate why the patterns could not be learned. Your report should be brief but different aspects of each case be explained clearly and completely.


