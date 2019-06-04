import numpy as np
from random import shuffle, seed
from time import perf_counter

# Convenience funtion to get the upcoming event
def get_next_event(events_list):
    event_times = [x[0] for x in events_list]
    t = min(event_times)
    return event_times.index(t)


def model(
        time_horizon,
        bartenders,  # Each bartender is represented as a list containing T/F (Female/Male)
        customer_lambda,  # mins
        p_drink,
        serve_time,  # mins
        flirt_time,  # mins
        drink_time,  # mins
        drink_price,  # $
        avg_tip,  # $
        patience_threshold,  # mins
        pmin_queue_shootout,
        p_queue_shootout,
        p_pianist_killed,
        pianist_net_worth,  # $
        shootout_loss,  # $
        poker_table_size,  # people
        poker_length,  # mins
        p_leave,
        p_lost_everything,
        p_jackpot,  # This is cumulative, including p_lost_everything
):
    bartenders_in_simulation = [[bartender, 0] for bartender in bartenders]
    # Each bartender is represented as a list containing T/F (Female/Male) and
    # a numeric variable indicating the time when the next action is finished

    sheriff_entry = np.random.uniform() * time_horizon
    events = [(np.random.exponential(customer_lambda), 'Customer_choice', 'new'),
              (sheriff_entry, 'Sheriff_entry'), (sheriff_entry + 60, 'Sheriff_exit'),
             (240, 'Lambda_down')]
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
                    events.append((clock[0] + np.random.exponential(customer_lambda),
                                   'Customer_choice', 'new'))
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
            shuffle(bartenders_in_simulation)  # works inplace
            for x in bartenders_in_simulation:
                if x[1] <= clock[0]:
                    # A free bartender available, customer is served
                    # Potentially check whether an if or random number generation is faster here
                    duration = x[0] * flirt_time * np.random.uniform() + serve_time
                    x[1] = clock[0] + duration
                    revenue += drink_price + np.random.gamma(shape=5, scale=avg_tip / 5) * x[0]
                    events.append((clock[0] + duration + np.random.exponential(drink_time),
                                   'Customer_choice', 'existing'))
                    waiting_time = False
                    break
                else:
                    waiting_time = min(waiting_time, x[1])
                    # Can we handle waiting within the same loop without creating an extra iteration?

            # If no bartender, wait at the bar
            if waiting_time:
                # Calculate cumulative waiting time
                try:
                    time_in_queue = clock[2] + waiting_time - clock[0]
                except IndexError:
                    time_in_queue = waiting_time - clock[0]
                # Determine if client gets nervous
                if time_in_queue > patience_threshold:
                    if np.random.uniform() < max(pmin_queue_shootout, clock[0] / 600 * p_queue_shootout):
                        events.append((clock[0] + patience_threshold, 'Shootout'))
                else:
                    events.append((waiting_time, 'Customer_drinks', time_in_queue))

        elif clock[1] == 'Sheriff_entry':
            sheriff_present = True

        elif clock[1] == 'Sheriff_exit':
            sheriff_present = False

        elif clock[1] == 'Shootout':
            if not sheriff_present:
                event_history.append(clock)
                revenue -= shootout_loss * max(1, np.log(clock[0]) / np.log(420)) + (
                            np.random.uniform() < p_pianist_killed) * pianist_net_worth
                return revenue, event_history
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
            poker_table = np.random.randint(poker_table_size)
            leavers = np.random.binomial(poker_table_size - poker_table, p_leave)
            customer_count -= leavers
            if (poker_table_size - poker_table - leavers) > 0:
                events.append(
                    (clock[0], 'Customer_drinks') * (poker_table_size - poker_table - leavers))
        
        elif clock[1] == 'Lambda_down':
            customer_lambda -= 5 # Avg time between new customers 5 mins shorter

        event_history.append(clock)
        # Get next event
        clock = events.pop(get_next_event(events))

    return revenue, event_history


def run_simulation(
        n_simulations,
        time_horizon=10 * 60,
        bartenders=(False, True),
        customer_lambda=25,
        p_drink=0.9,
        serve_time=5,
        flirt_time=15,
        drink_time=35,
        drink_price=2,
        avg_tip=1,
        patience_threshold=15,
        pmin_queue_shootout=0.03,
        p_queue_shootout=0.06,
        p_pianist_killed=0.05,
        pianist_net_worth=450,
        shootout_loss=200,
        poker_table_size=5,
        poker_length=10,
        p_leave=0.1,
        p_lost_everything=0.02,
        p_jackpot=0.04,
):
    revenues = np.zeros(n_simulations)
    event_histories = []
    for simulation_number in range(n_simulations):
        result, history = \
            model(
                time_horizon=time_horizon,
                bartenders=bartenders,
                customer_lambda=customer_lambda,
                p_drink=p_drink,
                serve_time=serve_time,
                flirt_time=flirt_time,
                drink_time=drink_time,
                drink_price=drink_price,
                avg_tip=avg_tip,
                patience_threshold=patience_threshold,
                pmin_queue_shootout=pmin_queue_shootout,
                p_queue_shootout=p_queue_shootout,
                p_pianist_killed=p_pianist_killed,
                pianist_net_worth=pianist_net_worth,
                shootout_loss=shootout_loss,
                poker_table_size=poker_table_size,
                poker_length=poker_length,
                p_leave=p_leave,
                p_lost_everything=p_lost_everything,
                p_jackpot=p_jackpot,
            )
        revenues[simulation_number] = result
        event_histories.append(history)

    return revenues, event_histories


# ## Scenariusz bazowy
# In[]:

results_normal, _ = run_simulation(n_simulations=1000)

# In[]:

plot_saving_mode = True

# In[]:


print("średni dochód: " + str(results_normal.mean()))
print("odchylenie z dochodu: " + str(results_normal.std()))


# In[]:


plt.figure(figsize = (8, 4))
plt.hist(results_normal, bins=100, density = True, color = 'blue')
plt.ylabel("Częstość")
plt.xlabel("Przychód [$]")
if plot_saving_mode:
    plt.savefig('raport/wykresy/histogram.pdf')
else:
    plt.show()


# ## Liczba zatrudnionych barmanów
# In[]:


def find_opt_solution(max_male, max_female, params = {'n_simulations': 200}):
    bartender_results = np.zeros((max_male+1, max_female+1))
    
    for i in tqdm.tqdm( product( np.arange(max_male+1), np.arange(max_female+1) ) ):
            # creating unique combination of male and female bartenders
            params['bartenders'] = [False] * i[0] + [True] * i[1]
            #running simulation
            results, _ = run_simulation(**params)
            #appending results
            bartender_results[i] = results.mean()
            
    path_matrix = np.zeros((max_male + 1, max_female + 1))
    optimum = 0
    diags = [bartender_results[::-1,:].diagonal(i) for i in range(1-bartender_results.shape[0], bartender_results.shape[1])]
    
    for i, x in enumerate(diags):
        i0 = min(i, path_matrix.shape[0]-1) - x.argmax()
        i1 = i-i0
        path_matrix[i0, i1] = x.max() - optimum
        optimum = x.max()
    return bartender_results, path_matrix

# In[]:

bartender_results, path_matrix = find_opt_solution(10, 10, {'n_simulations': 1000})

# In[]:


plt.figure(figsize = (8, 5))
sns.heatmap(bartender_results)
plt.ylabel("Liczba zatrudnionych barmanów płci męskiej")
plt.xlabel("Liczba zatrudnionych barmanów płci żeńskiej")
if plot_saving_mode:
    plt.savefig('raport/wykresy/barmani.pdf')
else:
    plt.show()

# In[]:


plt.figure(figsize = (8,5))
sns.set(style = 'whitegrid')
sns.heatmap(path_matrix, cmap = 'seismic_r', center = 0)
plt.ylabel("Liczba zatrudnionych barmanów płci męskiej")
plt.xlabel("Liczba zatrudnionych barmanów płci żeńskiej")
if plot_saving_mode:
    plt.savefig('raport/wykresy/opt_sciezka.pdf')
else:
    plt.show()

# ## Strategia cenowa
# In[]:


results_expensive, _ = run_simulation(n_simulations=n_simulations, drink_price=4, patience_threshold=10)
results_cheap, _ = run_simulation(n_simulations=n_simulations)
results_super_cheap, _ = run_simulation(
    n_simulations=n_simulations,
    drink_price=1,
    patience_threshold=20,
    customer_lambda=10)


# In[]:


print("średnia:")
print("droższe drinki: " + str(results_expensive.mean()))
print("tańsze drinki: " + str(results_cheap.mean()))
print("super tanie drinki: " + str(results_super_cheap.mean()))

print("\nodchylenie:")
print("droższe drinki: " + str(results_expensive.std()))
print("tańsze drinki: " + str(results_cheap.std()))
print("super tanie drinki: " + str(results_super_cheap.std()))


# In[]:


plt.figure(figsize=(8, 4))
df_to_plot = pd.DataFrame(
    {
        "Strategia cenowa salonu": 
            ["normalne ceny"] * n_simulations + 
            ["niskie ceny"] * n_simulations + 
            ["wysokie ceny"] * n_simulations,
        "Przychód [$]": np.concatenate((results_cheap, results_super_cheap, results_expensive), axis=0)
    }
)
ax = sns.barplot(x="Strategia cenowa salonu", y="Przychód [$]", data=df_to_plot, ci="sd")
if plot_saving_mode:
    plt.savefig('raport/wykresy/drinki.pdf')
else:
    plt.show()


# # Więlkosc stolow do pokera
# In[]:


results_normal, _ = run_simulation(n_simulations=n_simulations)
results_1, _ = run_simulation(n_simulations=n_simulations, poker_table_size=6, poker_length=15)
results_2, _ = run_simulation(n_simulations=n_simulations, poker_table_size=7, poker_length=20)
results_3, _ = run_simulation(n_simulations=n_simulations, poker_table_size=8, poker_length=25)


# In[]:


plt.figure(figsize=(8,4))
df_to_plot = pd.DataFrame(
    {
        "Stół do pokera": 
            ["na 5 graczy"] * n_simulations + ["na 6 graczy"] * n_simulations + ["na 7 graczy"] * n_simulations + ["na 8 graczy"] * n_simulations,
        "Przychód [$]": np.concatenate((results_normal, results_1, results_2, results_3), axis=0)
    }
)
sns.barplot(x="Stół do pokera", y="Przychód [$]", data=df_to_plot, ci="sd")
if plot_saving_mode:
    plt.savefig('raport/wykresy/poker.pdf')
else:
    plt.show()


# In[]:


print("średni przychód normal: " + str(results_normal.mean()))
print("średni przychód 1: " + str(results_1.mean()))
print("średni przychód 2: " + str(results_2.mean()))
print("średni przychód 3: " + str(results_3.mean()))

print("odchylenie normal: " + str(results_normal.std()))
print("odchylenie przychód 1: " + str(results_1.std()))
print("odchylenie przychód 2: " + str(results_2.std()))
print("odchylenie przychód 3: " + str(results_3.std()))








# # Analiza wrażliwości

# ## Zatrudnianie ładniejszych kelnerek

# In[]:

n_simulations = 1000
bartenders_opt = [True, True, True]

# In[]:

results_normal, _ = run_simulation(n_simulations=n_simulations,
                                   bartenders = bartenders_opt)
results_beautiful, _ = run_simulation(n_simulations=n_simulations,
                                      bartenders = bartenders_opt,
                                      flirt_time=25,
                                      avg_tip=5)


# In[]:

plt.figure(figsize=(8,4))
df_to_plot = pd.DataFrame(
    {
        "Personel": 
            ["ładny"] * n_simulations + ["ładniejszy"] * n_simulations,
        "Przychód [$]": np.concatenate((results_normal, results_beautiful), axis=0)
    }
)
ax = sns.barplot(x="Personel", y="Przychód [$]", data=df_to_plot, ci="sd")
if plot_saving_mode:
    plt.savefig('raport/wykresy/personel.pdf')
else:
    plt.show()


# In[]:


print("średni przychód ładna: " + str(results_normal.mean()))
print("średni przychód ładniejsza: " + str(results_beautiful.mean()))

print("odchylenie ładna: " + str(results_normal.std()))
print("odchylenie ładniejsza: " + str(results_beautiful.std()))

# In[ ]:

bartender_results_pretty, path_matrix_pretty = find_opt_solution(10, 10, {'n_simulations': n_simulations,
                                                           'flirt_time':25, 'avg_tip':5})


# In[ ]:

plt.figure(figsize = (8, 5))
sns.heatmap(bartender_results)
plt.ylabel("Liczba zatrudnionych barmanów płci męskiej")
plt.xlabel("Liczba zatrudnionych barmanów płci żeńskiej")
if plot_saving_mode:
    plt.savefig('raport/wykresy/barmani_ladni.pdf')
else:
    plt.show()

# In[ ]:

plt.figure(figsize = (8,5))
sns.heatmap(path_matrix, cmap = 'seismic_r', center = 0)
plt.ylabel("Liczba zatrudnionych barmanów płci męskiej")
plt.xlabel("Liczba zatrudnionych barmanów płci żeńskiej")
if plot_saving_mode:
    plt.savefig('raport/wykresy/opt_sciezka_ladni.pdf')
else:
    plt.show()

# ## Próg cierpliwosci klientow
# In[]:


patience_results = np.zeros(11)
patience_std = np.zeros(11)

for patience in tqdm.tqdm(range(len(patience_results))):
    results, _ = run_simulation(n_simulations=int(n_simulations), patience_threshold=patience)
    patience_results[patience] = results.mean()
    patience_std[patience] = results.std()


# In[]:


plt.figure(figsize = (8,4))
plt.fill_between(np.arange(0, 11), y1 = patience_results - patience_std,
                 y2 = patience_results + patience_std,
                alpha = 0.3)
plt.plot(patience_results)
plt.hlines(y = 0, xmin = 0, xmax = 10, linestyle = 'dashed')
plt.xlabel('Próg cierpliwości [min]')
plt.ylabel('Średni zysk baru [$]')
if plot_saving_mode:
    plt.savefig('raport/wykresy/zajecie_w_kolejce.pdf')
else:
    plt.show()


# ## Lepszy jakosciowo wystrój

# In[]:


decor_results = np.zeros(20)
decor_ccount = np.zeros(20)

for decor in tqdm.tqdm(range(1, len(decor_results))):
    results, histories = run_simulation(
        n_simulations=n_simulations,
        bartenders = bartenders_opt,
        shootout_loss=20*decor, 
        customer_lambda=(45 - 2*decor)) 
    decor_results[decor] = results.mean()
    decor_ccount[decor] = np.mean([[x[1:3] for x in history].count(('Customer_choice', 'new')) for history in histories])


# In[]:


plt.figure(figsize = (8,4))
plt.plot(decor_results)
plt.xlabel('Jakość wystroju')
plt.ylabel('Średni zysk baru [$]')
plt.hlines(y = 0, xmin = 0, xmax = 20, linestyle = 'dashed')
if plot_saving_mode:
    plt.savefig('raport/wykresy/wystroj.pdf')
else:
    plt.show()


# # Unused scenarios
# # Strzelaniny
# In[]:


results_faster_shootout, _ = run_simulation(n_simulations=n_simulations, p_lost_everything=0.05)
results_normal_shootout, _ = run_simulation(n_simulations=n_simulations)
results_slower_shootout, _ = run_simulation(n_simulations=n_simulations, p_lost_everything=0.01)


# In[]:


print("średnia:")
print("szybciej strzelaniny: " + str(results_faster_shootout.mean()))
print("normalne strzelaniny: " + str(results_normal_shootout.mean()))
print("wolniejsze strzelaniny: " + str(results_slower_shootout.mean()))


# In[]:


plt.figure(figsize=(8,4))
f, axes = plt.subplots(1, 1)
df_to_plot = pd.DataFrame(
    {
        "Prawdopodobieństwo strzelaniny": 
            ["niskie"] * n_simulations + 
            ["normalne"] * n_simulations + 
            ["wysokie"] * n_simulations,
        "Przychód [$]": np.concatenate((results_slower_shootout, results_normal_shootout, results_faster_shootout), axis=0)
    }
)
ax = sns.barplot(x="Prawdopodobieństwo strzelaniny", y="Przychód [$]", data=df_to_plot, ci="sd")
if plot_saving_mode:
    plt.savefig('raport/wykresy/p_strzelaniny.pdf')
else:
    plt.show()

