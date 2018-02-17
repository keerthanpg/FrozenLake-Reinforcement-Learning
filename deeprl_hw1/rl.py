# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """  
    policy=np.zeros(env.nS)    
    for i in range(env.nS):   
      max_action=-1
      max_value= -1
      for action in env.P[i]:
        value= value_func[env.P[i][action][j][1]]                      
        if value>=max_value:
          max_value = value
          max_action = action
      policy[i]=max_action
       
    return policy     
         
   


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    iteration = 0
    value_func=np.zeros(env.nS)
    
    delta=100
    while delta>tol:
      iteration+=1
      delta = 0
      v = np.copy(value_func)
       # initialises value function
      for i in range(env.nS):        
        j=policy[i] # checks what action policy says        
        value_func[i]=0
        for item in env.P[i][j]:          
          value_func[i]+=item[0]*(item[2]+gamma*v[item[1]]) #updates value function of current state        
        #check for terminal       
      delta = max(delta, max(np.absolute(value_func-v)))
     

    return value_func, iteration


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    iteration = 0
    value_func=np.zeros(env.nS)
    delta=100
    while delta>tol:
      iteration+=1
      delta = 0
      v = np.copy(value_func)
       # initialises value function
      for i in range(env.nS):        
        j=policy[i] # checks what action policy says        
        value_func[i]=0
        for item in env.P[i][j]:          
          value_func[i]+=item[0]*(item[2]+gamma*value_func[item[1]]) #updates value function of current state        
        #check for terminal       
      delta = max(delta, max(np.absolute(value_func-v)))
     

    return value_func, iteration


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    iteration = 0
    randperm=np.random.permutation(env.nS)
    value_func=np.zeros(env.nS)

    
    delta=100
    while delta>tol:
      iteration+=1
      delta = 0
      v = np.copy(value_func)
       # initialises value function
      for i in randperm:        
        j=policy[i] # checks what action policy says        
        value_func[i]=0
        for item in env.P[i][j]:          
          value_func[i]+=item[0]*(item[2]+gamma*value_func[item[1]]) #updates value function of current state        
        #check for terminal       
      delta = max(delta, max(np.absolute(value_func-v)))
     

    return value_func, iteration


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    return np.zeros(env.nS), 0


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True     
    for i in range(env.nS):      
      old_action = policy[i]
      max_action=-1
      max_value= -1
      for action in env.P[i]:
        value=0
        for j in range(len(env.P[i][action])):                                   
          value += env.P[i][action][j][0]*(env.P[i][action][j][2] + gamma* value_func[env.P[i][action][j][1]])              
        if value>max_value:
          max_value = value
          max_action = action
      policy[i]=max_action
      if old_action != policy[i]:
        policy_stable = False  
    return policy_stable, policy     
       



def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """

    

    policy = np.zeros(env.nS, dtype='int') #shouldnt actions be assigned randomly?
    value_func = np.zeros(env.nS)
    p_iterations = 0
    v_iterations = 0


    while True:
      #print("Evaluate policy", policy)
      value_func, iteration = evaluate_policy_sync(env, gamma, policy, max_iterations)
      #print("Updated value function", value_func)
      v_iterations+=iteration #how this one makes sense..
      #print("Improving policy")
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      #print("updated policy", policy, "stability:", policy_stable)
      p_iterations+=1
      if policy_stable==True:
        return policy, value_func, p_iterations, v_iterations
    
    


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int') #shouldnt actions be assigned randomly?
    value_func = np.zeros(env.nS)
    p_iterations = 0
    v_iterations = 0


    while True:
      #print("Evaluate policy", policy)
      value_func, iteration = evaluate_policy_async_ordered(env, gamma, policy, max_iterations)
      #print("Updated value function", value_func)
      v_iterations+=iteration #how this one makes sense..
      #print("Improving policy")
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      #print("updated policy", policy, "stability:", policy_stable)
      p_iterations+=1
      if policy_stable==True:
        return policy, value_func, p_iterations, v_iterations
    


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int') #shouldnt actions be assigned randomly?
    value_func = np.zeros(env.nS)
    p_iterations = 0
    v_iterations = 0


    while True:
      #print("Evaluate policy", policy)
      value_func, iteration = evaluate_policy_async_randperm(env, gamma, policy, max_iterations)
      #print("Updated value function", value_func)
      v_iterations+=iteration #how this one makes sense..
      #print("Improving policy")
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      #print("updated policy", policy, "stability:", policy_stable)
      p_iterations+=1
      if policy_stable==True:
        return policy, value_func, p_iterations, v_iterations

def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    return policy, value_func, 0, 0


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    iteration = 0
    value_func = np.zeros(env.nS)
    delta=100
    while delta>tol:
      iteration+=1
      #print("iteration number:", iteration)
      delta = 0
      v = np.copy(value_func)
      #print(v)
    
      for i in range(env.nS): 
        value_actions=np.zeros(env.nA)
        for j in range(env.nA):
          for item in env.P[i][j]:          
            value_actions[j]+=item[0]*(item[2]+gamma*v[item[1]]) #updates value function of current state  
        value_func[i]=max(value_actions)     
      #print(value_func )
      #print(v)
      delta = max(delta, max(np.absolute(value_func-v)))
    return value_func, iteration


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    iteration = 0
    value_func = np.zeros(env.nS)
    delta=100
    while delta>tol:
      iteration+=1
      #print("iteration number:", iteration)
      delta = 0
      v = np.copy(value_func)
      #print(v)
    
      for i in range(env.nS): 
        value_actions=np.zeros(env.nA)
        for j in range(env.nA):
          for item in env.P[i][j]:          
            value_actions[j]+=item[0]*(item[2]+gamma*value_func[item[1]]) #updates value function of current state  
        value_func[i]=max(value_actions)     
      #print(value_func )
      #print(v)
      delta = max(delta, max(np.absolute(value_func-v)))
    return value_func, iteration


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    iteration = 0
    value_func = np.zeros(env.nS)
    randperm=np.random.permutation(env.nS)
    delta=100
    while delta>tol:
      iteration+=1
      #print("iteration number:", iteration)
      delta = 0
      v = np.copy(value_func)
      #print(v)
    
      for i in randperm: 
        value_actions=np.zeros(env.nA)
        for j in range(env.nA):
          for item in env.P[i][j]:          
            value_actions[j]+=item[0]*(item[2]+gamma*value_func[item[1]]) #updates value function of current state  
        value_func[i]=max(value_actions)     
      #print(value_func )
      #print(v)
      delta = max(delta, max(np.absolute(value_func-v)))
    return value_func, iteration


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return np.zeros(env.nS), 0

