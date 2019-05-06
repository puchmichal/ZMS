# # ZMS LAB 043 - case GWINTEX
# 
# kontakt: annaszczurek2@gmail.com
# 
# 
# ## Opis zajęć
# 
# 
# Firma GWINTEX S.A. jest międzynarodowym potentatem w dziedzinie produkcji korkociągów. Korkociągi są wytwarzane na bardzo nowoczesnych maszynach metalurgicznych. W związku ze znacznym wzrostem zamówień firma planuje uruchomienie nowej hali produkcyjnej, w której znajdzie się **n=6 maszyn**. Do każdej maszyny jest przypisany operator, który jest odpowiedzialny za jej obsługę oraz usuwanie awarii. Na podstawie pomiarów historycznych wiadomo, że **czas bezawaryjnej pracy maszyny ma rozkład wykładniczy ze średnią 75 minut**. W przypadku wystąpienia awarii operator dzwoni do warsztatu z prośbą o dostarczenie pakietu narzędzi naprawczych. Pakiet narzędzi jest bardzo ciężki i w związku z tym musi być transportowany za pomocą przenośnika taśmowego (taśmociągu). **Czas transportu zestawu narzędzi do maszyny wynosi *ti*, i=1..6. Czas naprawy jest zmienną losową z rozkładu Erlanga k=3 i średnio wynosi 15 minut**. Po ukończeniu naprawy narzędzia są powtórnie umieszczane na taśmociągu i wracają w komplecie do warsztatu celem ich uzupełnienia. Ze względu na specyfikę specjalistycznych narzędzi nie jest możliwe dokonywanie kolejnych napraw przed powrotem narzędzi do warsztatu. Ze względu na bardzo wysoką cenę jednego pakietu narzędzi naprawczych ich liczba ***m* jest mniejsza od liczby maszyn w hali produkcyjnej**. Gdy w danej chwili pakiet narzędzi nie jest dostępny operator czeka aż inny pakiet wróci do warsztatu i zostanie mu wysłany.
# 
# Zarząd firmy GWINTEX zastanawia się **jakie powinno być rozmieszczenie urządzeń na hali produkcyjnej** oraz **ile pakietów narzędziowych do obsługi maszyn należy zakupić**. Rozważane są dwie organizacje hali produkcyjnej – układ liniowy oraz układ gniazdowy. **W układzie liniowym czas transportu narzędzi z warsztatu do maszyny wynosi ***ti=i*2***, natomiast **w układzie gniazdowym czas ten jest stały i wynosi 3 minuty**. Czas transportu narzędzi do warsztatu jest taki sam jak czas transportu do maszyny. Wprowadzenie układu gniazdowego wiąże się z wyższymi kosztami instalacyjnymi związanymi z uruchomieniem sześciu niezależnych taśmociągów.
# 
# 
# 
# 
# ## ROZWIĄZANIE
# 
# Autorzy kodu: Anna Chojnacka, Michał Puchalski, Paweł Sadłowski
# 
# ### 1. Stałe i zmienne wykorzystane w modelu

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[32]:


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[2]:


# liczba maszyn
n = 6 

# średni czas pracy bez usterki
avg_working_time = 75 # minut

# średni czas naprawy
avg_repair_time = 15 # minut

# ilość zestawów narzędzi
m = 2

# horyzont analizy
horizon = 30 # dni

# liczba uruchomień symulacji
iterations = 1000   


# In[3]:


# Toolset transport times for all considered setups - this should facilitate easier handling of new
# setups
transport_times = {'L': {i:2*(i+1) for i in range(6)}, 'G': {i:3 for i in range(6)},
                   'L2': {0:2, 1:2, 2:4, 3:4, 4:6, 5:6}}


# ### 2. Model
# 
# Wektory, które mają za zadanie kontrolować stan symulacji:
# 
# - momenty wystąpienia kolejnych zdarzeń
# - status narzędzi i maszyn 
#     - `W` - pracuje 
#     - `Q` - czeka na narzedzia 
#     - `R` - jest naprawiona
# - czas ich bezczynności
# - events --> wektor zdarzeń, które zmieniają stan symulacji (np. zepsucie się maszyny, czas naprawy, itp.)

# In[4]:


def model(horizon, avg_working_time, avg_repair_time, n, m, setup):
    # setup - układ liniowy "L" lub gniazdowy "G"
    
    # horyzont działania w minutach
    horizon = horizon * 24 * 60 
    
    # wektor zdarzeń, który zmienia stan symulacji
    events = list(np.random.exponential(avg_working_time, n))
    
    # status - określa aktualny stan maszyny 
    status = ["W"] * n

    # t_start - określa początek bezczynności maszyny
    t_start = [0] * n

    # t_cum - skumulowany czas bezczynności maszyny
    t_cum = [0] * n

    # tools_loc lokalizacja narzedzi - albo numer maszyny albo -1 czyli warsztat
    tools_loc = [-1] * m

    # tools_occupied czas zajecia zestawu przez naprawianą maszynę
    tools_occupied = [0] * m
    
    # zegar symulacji- najblizsze zadanie, które ma być wykonane
    t = min(events)
    
    # rozpoczynamy symulacje "skacząc" po kolejnych zdarzeniach  
    while t <= horizon:
        
        # jeżeli zestawy nie są aktualnie zajęte to przenosimy je z powrotem do warsztatu
        for i in range(m):
            if tools_occupied[i] <= t:
                tools_loc[i] = -1

        # wybieramy maszynę, której dotyczy zdarzenie
        machine = events.index(t)
        
        """
        Gdy maszyna, której dotyczy zdarzenie ma status "W":
            - to najpierw zaktualizuj wektor t_start dla tej maszyny jako początek jej bezczynności = t.
            - następnie sprawdź czy dostępny jest jakiś zestaw naprawczy. Jezeli nie:
                - to ustaw status maszyny na "Q" 
                - zaktualizuj wektor events podajac mu najkrótszy czas oczekiwania na wolny zestaw.
              Jeżeli tak:
                - ustaw status maszyny na "R"
                - wyznacz czas  potrzebny na naprawę maszyny w zależności od ukladu taśmociągu 
                (czas transportu + czas naprawy)
                - ustaw koniec naprawy jako zdarzenie dla danej maszyny
                - zaktualizuj wektor tools_loc dla odpowiedniego zestawu podając numer maszyny, którą on obsługuje
                - zaktualizuj wektor tools_occupied czasem jaki mu to zajmie (2* transport + naprawa)
        """
        if status[machine] == "W":
            t_start[machine] = t
            tools = - 1
            for i in range(m):
                if tools_loc[i] == -1:
                    tools = i
                    break
            if tools == -1 :
                status[machine] = "Q"
                events[machine] = min(tools_occupied)
            else:
                status[machine] = "R"
                transport_time = transport_times[setup][machine]
                repair_time = np.random.gamma(3, avg_repair_time/3)
                events[machine] += repair_time + transport_time
                tools_loc[tools] = machine
                tools_occupied[tools] += repair_time + 2 * transport_time
                
                """
                Gdy maszyna ma status "Q":
                    - wybierz dostępny zestaw naprawczy
                    - ustal status maszyny na "R"
                    - zaktualizuj wektor tools_loc lokalizacją narzedzi i tools_occupied 
                    czasem jaki zajmie ich transport (w dwie strony) i naprawa maszyny
                    -zaktualizuj wektor zdarzeń czasem potrzebnym na naprawę maszyny i transport narzedzi
                """
                
        elif status[machine] == "Q":
            for i in range(m):
                if tools_loc[i] == -1:
                    tools = i
                    break
            status[machine] = "R"
            transport_time = transport_times[setup][machine]
            repair_time = np.random.gamma(3, avg_repair_time/3)
            events[machine] += repair_time + transport_time
            tools_loc[tools] = machine
            tools_occupied[tools] += repair_time + 2 * transport_time 
            """
            Gdy maszyna ma status "R":
                - ustal jej status na "W"
                - wyznacz czas kolejnej awarii i zaktualizuj wektor events
                - wylicz czas bezczynnosci i uzupelnij o niego liste t_cum
            """
            
        else:
            status[machine] = "W"
            events[machine] += np.random.exponential(avg_working_time)
            t_cum[machine] += t - t_start[machine]
        
        # ustalamy nowe t
        t = min(events)
        
    # wynik - liste skumulowanych bezczynnosci dla kazdej z maszyn
    return (t_cum)


# ### 3. Funkcja do uruchomienia symulacji

# Added seed to provide reproducibility of results

# In[5]:


def run_model (iterations, horizon, avg_working_time, avg_repair_time, n, m, setup):
    avg_t_cum = []
    for i in range (iterations):
        np.random.seed(i)
        avg_t_cum.append(model( horizon, avg_working_time, avg_repair_time, n, m, setup))
    return list(map(np.mean, np.transpose(avg_t_cum)))


# ### 3. Simplest comparison

# In[10]:


import time


# In[11]:


results = {}
for setup in ['L', 'G', 'L2']:
    for m in range(5):
        results[(setup, m+1)] = run_model(iterations, horizon, avg_working_time, avg_repair_time, n, m+1, setup)


# In[12]:


df = pd.DataFrame.from_dict(results).T


# In[16]:


df2 = df.mean(axis=1).unstack()


# In[17]:


df2 


# In[18]:


df2.loc["L2",:] / df2.loc['L',:]


# In[21]:


df2.loc[:,2:5].values / df2.loc[:,1:4].values


# In[22]:


(df2.loc[:,1].values - df2.loc[:,2].values) / 60


# In[23]:


(df2.loc['L':'L2',2].values - df2.loc['G',2]) / 60


# ### 5. Sensitivity analysis

# #### Repair time reduction

# Assuming 2 toolkits in each setup

# In[26]:


results_rapair = {}
for setup in ['L', 'G', 'L2']:
    for rapair_improvement in range(5):
        results_rapair[(setup, rapair_improvement)] = run_model(iterations, horizon, avg_working_time, avg_repair_time - rapair_improvement, n, m, setup)


# In[27]:


pd.DataFrame.from_dict(results_rapair)


# In[28]:


df_repair = pd.DataFrame.from_dict(results_rapair).T


# In[31]:


np.mean(df_repair.loc['G'].iloc[1:,:].values - df_repair.loc['G'].iloc[:4,:].values)/60


# In[34]:


np.mean(df_repair.loc['L'].iloc[1:,:].values - df_repair.loc['L'].iloc[:4,:].values)/60


# In[37]:


np.mean(df_repair.loc['L2'].iloc[1:,:].values - df_repair.loc['L2'].iloc[:4,:].values)/60


# In[36]:


fig, ax = plt.subplots(1, 3, figsize = (10, 3), sharey = True)
for x in zip(ax, ['L', 'L2', 'G']):
    df_repair.xs(axis=0, drop_level=True, key=x[1], level=0).plot(ax = x[0])
    x[0].set_title('Układ '+{'L':'liniowy', 'L2':'liniowy zmodyfikowany', 'G': 'gniazdowy'}[x[1]])
ax[1].set_xlabel('Spadek średniego czasu naprawy [min]')
ax[0].set_ylabel('Przeciętny czas przestoju [min]')
plt.savefig('raport/wykresy/szkolenia.pdf', bbox_inches = 'tight')


# #### Temporary machine shutdown

# In[40]:


results_machine_reduction = {}
for setup in ['L', 'G', 'L2']:
    for n in range(6):
        n_machines = n + 1
        results_machine_reduction[(setup, n_machines)] = run_model(iterations, horizon, avg_working_time, avg_repair_time, n_machines, m, setup)


# In[41]:


df_machine_reduction = pd.DataFrame.from_dict(results_machine_reduction, orient='index')


# In[42]:


df_machine_reduction


# In[43]:


df_comparison = pd.DataFrame()


# In[44]:


df_machine_reduction.iloc[:6,:]


# In[46]:


df_comparison["L"] = df_machine_reduction.iloc[:6,:].mean(axis=1).values


# In[48]:


df_comparison["G"] = df_machine_reduction.iloc[6:12,:].mean(axis=1).values


# In[50]:


df_comparison["L2"] = df_machine_reduction.iloc[12:,:].mean(axis=1).values


# In[51]:


df_comparison = df_comparison.set_index(np.arange(1, 7))


# In[41]:


df_comparison.plot(figsize = (5,3), legend=False)
plt.xlabel('Liczba działających maszyn')
plt.ylabel('Przeciętny czas przestoju [min]')
plt.legend(['liniowy', 'gniazdowy', 'liniowy zmod.'])
plt.savefig('raport/wykresy/recesja.pdf', bbox_inches = 'tight')


# #### Longer time between failures

# In[63]:


results_better_machines = {}
for setup in ['L', 'G', 'L2']:
    for working_improvement in range(15):
        results_better_machines[(setup, working_improvement)] = run_model(iterations, horizon, avg_working_time + working_improvement, avg_repair_time, n_machines, m, setup)


# In[83]:


df_better_machines = pd.DataFrame.from_dict(results_better_machines).T


# In[84]:


df_better_machines


# In[44]:


fig, ax = plt.subplots(1, 3, figsize = (10,3), sharey = True)
for x in zip(ax, ['L','L2','G']):
    df_better_machines.xs(axis=0, drop_level=True, key=x[1], level=0).plot(ax = x[0])
    x[0].set_title('Układ '+{'L':'liniowy', 'L2':'liniowy zmodyfikowany', 'G': 'gniazdowy'}[x[1]])
ax[0].set_ylabel('Przeciętny czas przestoju [min]')
ax[1].set_xlabel('Wzrost średniego czasu bezawaryjnej pracy [min]')
plt.savefig('raport/wykresy/bezawaryjna_praca.pdf', bbox_inches = 'tight')


# In[46]:


df_better_machines.mean(axis = 1).groupby(level = 0).apply(lambda x: (x.iloc[-1] - x.iloc[0])/(len(x)-1))

