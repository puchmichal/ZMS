# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:40:44 2019

@author: admin
"""

# %%

import numpy as np
from random import shuffle
from time import perf_counter

# %%

# Convenience funtion to get the upcoming event
def getNextEvent(events):
    event_times = [x[0] for x in events]
    t = min(event_times)
    return event_times.index(t)
# %%

time_horizon = 24*60 # mins
bartenders = [[False, 0], [True, 0]]
# Each bartender is represented as a list containing T/F (Female/Male) and
# a numeric variable indicating the time when the next action is finished

customer_lambda = 25 # mins
p_drink = 0.9
serve_time = 5 # mins
flirt_time = 15 # mins
drink_time = 35 # mins
drink_price = 2 # $
avg_tip = 5 # $
patience_threshold = 15 # mins
p_queue_shootout = 0.01
p_pianist_killed = 0.03
pianist_net_worth = 100 # $
shootout_loss = 50 # $
poker_table_size = 5 # people
poker_length = 10 # mins
p_leave = 0.1
p_lost_everything = 0.05
p_jackpot = 0.1 # This is cumulative, including p_lost_everything

# %%

def model(time_horizon, bartenders):
    sheriff_entry = np.random.uniform() * time_horizon    
    events = [(np.random.exponential(customer_lambda), 'Customer_choice', 'new'),
              (sheriff_entry, 'Sheriff_entry'), (sheriff_entry + 60, 'Sheriff_exit')]
    revenue = 0    
    customer_count = 0
    poker_table = 0
    event_history = []
    clock = (0, 'start')
    sheriff_present = False
    
    # The event loop
    while clock[0] < time_horizon:
        
        # Check event type
        if clock[1] == 'Customer_choice':
            # Does he stay in the saloon?
            if (clock[2] == 'existing') & (np.random.uniform() < p_leave):
                customer_count -= 1
            else:
                if clock[2] == 'new':
                    # Increase guest count
                    customer_count += 1
                    # Generate next customer
                    events.append((clock[0] + np.random.exponential(customer_lambda), 'Customer_choice', 'new'))
                # Choose action
                if poker_table < poker_table_size:
                    if np.random.uniform() < p_drink:
                        events.append((clock[0], 'Customer_drinks'))
                    else:
                        poker_table += 1
                        if poker_table == poker_table_size:
                            events.append((clock[0] + poker_length, 'Poker_finish'))
                else:
                    events.append((clock[0], 'Customer_drinks'))
        
        if clock[1] == 'Customer_drinks':
            waiting_time = np.inf
            shuffle(bartenders) # works inplace
            for x in bartenders:
                if x[1] <= clock[0]:
                    # A free bartender available, customer is served
                    # Potentially check whether an if or random number generation is faster here
                    duration = x[0] * flirt_time * np.random.uniform() + serve_time
                    x[1] = clock[0] + duration
                    revenue += drink_price + np.random.gamma(shape = 5, scale = avg_tip / 5) * x[0]
                    events.append((clock[0] + duration + np.random.exponential(drink_time),
                       'Customer_choice', 'existing'))
                    waiting_time = False
                    break
                else:
                    waiting_time = min(waiting_time, x[1])
                    # Can we handle waiting within the same loop without creating an extra iteration?
            
            # If no bartender, wait in line
            if waiting_time:
                if waiting_time - clock[0] > patience_threshold:
                    if np.random.uniform() < p_queue_shootout:
                        events.append((clock[0] + patience_threshold, 'Shootout'))
                else:
                    events.append((waiting_time, 'Customer_drinks'))
        
        elif clock[1] == 'Sheriff_entry':
            sheriff_present = True
        
        elif clock[1] == 'Sheriff_exit':
            sheriff_present = False
            
        elif clock[1] == 'Shootout':
            if not sheriff_present:
                revenue -= shootout_loss + (np.random.uniform() < p_pianist_killed) * pianist_net_worth
                return (revenue, event_history)
        # Decide on scenario handling here
        
        elif clock[1] == 'Poker_finish':
            outcome = np.random.uniform()
            if outcome < p_lost_everything:
                # The loser starts a shootout
                events.append((clock[0], 'Shootout'))
            elif outcome < p_jackpot:
                # The winner buys everybody a round
                revenue += customer_count * drink_price
            # Some players stay, some leave, some grab a drink
            poker_table = np.random.randint(poker_table_size - 1)
            leavers = np.random.binomial(poker_table_size - poker_table, p_leave)
            events.append((clock[0], 'Customer_drinks') * (poker_table_size - poker_table - leavers))
        
        event_history.append(clock)
        # Get next event
        clock = events.pop(getNextEvent(events))
    
    return (revenue, event_history)
        
# %% Test run of the model

t0 = perf_counter()
wynik = model(time_horizon, bartenders)
print('Time elapsed: {:f}'.format(perf_counter() - t0))


# %%
exec_times = []
for i in range(50):
    t0 = perf_counter()
    model(time_horizon, bartenders)
    exec_times.append(perf_counter() - t0)
print('Average execution time {:f} with a std of {:f}'.format(np.mean(exec_times), np.std(exec_times)))

# %% Timing

# This thing throws an unexplained error at the moment, which didn't occur when the above was used
# Function seems pretty fast anyway, 1k iterations shouldn't take longer than approx. 6s if it scales linearly
%timeit model(time_horizon, bartenders)