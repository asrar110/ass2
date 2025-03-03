#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import math
import random
def tanh(x):
  return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
def tanh_derivative(x):
  return 1 - tanh(x)**2
input_size = 2
hidden_size = 2
output_size = 1
W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
W2 = [[random.uniform(-0.5, 0.5)] for _ in range(hidden_size)]
b1 = [0.5, 0.5]
b2 = [0.7]
input_data = [0.8, 0.3]
learning_rate = 0.1
def forward_propagation(input_data, W1, b1, W2, b2):
  hidden_input = [0] * hidden_size
  hidden_output = [0] * hidden_size
  output_input = [0] * output_size
  output = [0] * output_size
  for i in range(hidden_size):
    for j in range(input_size):
      hidden_input[i] += input_data[j] * W1[j][i]
    hidden_input[i] += b1[i]
    hidden_output[i] = tanh(hidden_input[i])
  for i in range(output_size):
    for j in range(hidden_size):
      output_input[i] += hidden_output[j] * W2[j][i]
    output_input[i] += b2[i]
    output[i] = tanh(output_input[i])
  return hidden_input, hidden_output, output_input, output
def backward_propagation(input_data, expected_output, hidden_input, hidden_output, output_input, output, W1, W2):
  output_error = [expected_output[i] - output[i] for i in range(output_size)]
  output_delta = [output_error[i] * tanh_derivative(output_input[i]) for i in range(output_size)]
  hidden_error = [0] * hidden_size
  for i in range(hidden_size):
    for j in range(output_size):
      hidden_error[i] += output_delta[j] * W2[i][j]
  hidden_delta = [hidden_error[i] * tanh_derivative(hidden_input[i]) for i in range(hidden_size)]
  for i in range(output_size):
    b2[i] += learning_rate * output_delta[i]
    for j in range(hidden_size):
      W2[j][i] += learning_rate * output_delta[i] * hidden_output[j]
  for i in range(hidden_size):
    b1[i] += learning_rate * hidden_delta[i]
    for j in range(input_size):
      W1[j][i] += learning_rate * hidden_delta[i] * input_data[j]
expected_output = [0.7]  
hidden_input, hidden_output, output_input, output = forward_propagation(input_data, W1, b1, W2, b2)
backward_propagation(input_data, expected_output, hidden_input, hidden_output, output_input, output, W1, W2)
print( output)


# In[ ]:




