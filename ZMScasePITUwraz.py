
# coding: utf-8

# # ZMS LAB 02 - case PITU
# 
# kontakt: annaszczurek2@gmail.com

# ## Kiedy ruina = RUINA ?
# 
# *opr. P. Wojewnik na podstawie:*
# 
# *„Optymalizacja portfela szkód” K. Saduś, M. Kwiecień, R. Lipiński* oraz *„Ubezpieczenia komunikacyjne firmy ABC” A. Kołota, M. Mączyńska*
# 
# 
# Prezes zarządu zakładu ubezpieczeń PiTU S.A., Marcin R., zasłyszał, że Prezydent planuje naturalizować wszystkich chętnych z Dżydżykistanu. *No to teraz się zacznie*, pomyślał, *Dżydżykowie jeżdżą jak wariaci, wzrośnie szkodowość, a więc musimy podnieść dla nich ceny.* 
# 
# Marcin R. dzwoni do aktuariuszki – Aldony P., ale tu czeka go niespodzianka. *Mój synek zachorował... Grypa to nic wielkiego... Ale w świńskim wydaniu bywa niebezpieczna.* Marcin z bólem serca odsyła Aldonę do domu, a zlecenie dostaje Marek T., asystent Aldony. Pytanie brzmi: **czy składka 500 zł jest wystarczająca ?**
# 
# Dotychczasowe doświadczenia PiTU S.A. z Dżydżykami są następujące:
# 
# <table>
# <tr></tr>
# <tr><td>
# 
# | Liczba szkód    | Liczba polis
# |-----------------|-------------
# | 0               | 3 437
# | 1               | 522
# | 2               | 40
# | 3               | 2
# | 4               | 0
# | 5               | 0
# 
# </td><td>
# 
# | Wielkość szkody | Liczba szkód
# |-----------------|-------------
# | -               | 0
# | 100             | 0
# | 200             | 2
# | 500             | 27
# | 1 000           | 52
# | 2 000           | 115
# | 5 000           | 203
# | 10 000          | 106
# | 20 000          | 42
# | 40 000          | 14
# | 50 000          | 0
# | 55 000          | 0
# | 60 000          | 1
# 
# </td></tr> </table>
# 
# Aldona rzuciła jeszcze przez telefon, że **liczbę szkód dobrze opisuje rozkład Poissona**, natomiast **wielkość szkód – rozkład log-normalny**. Z Marcina udało się wydusić **oczekiwaną liczbę klientów – 100** – oraz **aktualną nadwyżkę 10 000**.
# 
# 
# **Pytania:**
# 1.	Jaką ustalić składkę OC, aby ruina kierowców nie była udziałem PiTU S.A.?
# 2.	Czy nadwyżka końcowa będzie równa początkowej?
# 3.	Jakie jest zagrożenie ruiną?
# 4.	Jaka powinna być nadwyżka i składka, żeby prawdopodobieństwo ruiny było mniejsze niż 0,01?
# 

# ## ROZWIĄZANIE
# 
# opr. P

# In[ ]:


path = r'C:\Users\Nusza\Desktop\ZMS - push'

import csv
import scipy as sc
import matplotlib.pyplot as plt
from scipy.stats.stats import kstest
import numpy as np
from pandas import DataFrame


# ## 1. Dane - rozkłady, wyznaczanie parametrów
# 
# ### Liczba szkód

# In[ ]:


liczba_szkod = {0 : 3437, 
                1 : 522, 
                2 : 40, 
                3 : 2, 
                4 : 0, 
                5 : 0}

plt.bar(list(liczba_szkod.keys()), 
        list(liczba_szkod.values()))


# In[ ]:

SREDNIA_LICZBA_SZKOD = (sum([x * y for x, y in liczba_szkod.items()]) / 
                        sum(liczba_szkod.values()))

SREDNIA_LICZBA_SZKOD1 = ((sum([x * y for x, y in liczba_szkod.items()])*1.05) / 
                        sum(liczba_szkod.values()))

SREDNIA_LICZBA_SZKOD2 = ((sum([x * y for x, y in liczba_szkod.items()])*0.95) / 
                        sum(liczba_szkod.values()))

# czy liczba szkód ma faktycznie rozklad Poissona?
poisson_test0 = [sc.stats.poisson.pmf(i, (SREDNIA_LICZBA_SZKOD)) * 
                sum(liczba_szkod.values()) for i in range(len(liczba_szkod))]

poisson_test1 = [sc.stats.poisson.pmf(i, (SREDNIA_LICZBA_SZKOD1)) * 
                sum(liczba_szkod.values()) for i in range(len(liczba_szkod))]

poisson_test2 = [sc.stats.poisson.pmf(i, (SREDNIA_LICZBA_SZKOD2)) * 
                sum(liczba_szkod.values()) for i in range(len(liczba_szkod))]

fig = plt.figure(figsize = (10,5))
plt.bar(list(liczba_szkod.keys()), poisson_test0, color = "orange")
plt.bar(list(liczba_szkod.keys()), list(liczba_szkod.values()), color = 'blue', width = 0.6)
plt.ylabel('Częstość')
plt.xlabel('Liczba zgłoszonych szkód')
plt.legend(['Rozkład empiryczny', 'Rozkład teoretyczny'])
fig.savefig(path+r'\rozklad_l_szkod.pdf')

fig = plt.figure(figsize = (10,5))
plt.bar(list(liczba_szkod.keys()), poisson_test1, color = "green", width = 0.8)
plt.bar(list(liczba_szkod.keys()), poisson_test2, color = "orange", width = 0.7)
plt.bar(list(liczba_szkod.keys()), list(liczba_szkod.values()), color = 'blue', width = 0.6)
plt.ylabel('Częstość')
plt.xlabel('Liczba zgłoszonych szkód')
plt.legend(['Rozkład empiryczny (niedoszacowany parametr)', 'Rozkład empiryczny (przeszacowany parametr)', 'Rozkład teoretyczny'])
fig.savefig(path+r'\rozklad_l_szkod_wraz.pdf')

# In[ ]:

# test chi-kwadrat z biblioteki scipy pomoże odpowiedziec na pytanie:
test0 = sc.stats.chisquare(list(liczba_szkod.values()), f_exp = poisson_test0)
if test0[1] > 0.05:
    print ("brak podstaw do odrzucenia hipotezy zerowej")
else:
    print("należy odrzucic hipoteze zerowa")

# test chi-kwadrat z biblioteki scipy pomoże odpowiedziec na pytanie:
test1 = sc.stats.chisquare(list(liczba_szkod.values()), f_exp = poisson_test1)
if test1[1] > 0.05:
    print ("brak podstaw do odrzucenia hipotezy zerowej")
else:
    print("należy odrzucic hipoteze zerowa")
    
test2 = sc.stats.chisquare(list(liczba_szkod.values()), f_exp = poisson_test2)
if test2[1] > 0.05:
    print ("brak podstaw do odrzucenia hipotezy zerowej")
else:
    print("należy odrzucic hipoteze zerowa")    


# ### Wielkość szkód

# In[ ]:


szkody = []
with open(path+'\\szkody.txt','r') as csvfile:
    reader = csv.reader (csvfile, delimiter=";")
    for row in reader:
        szkody.append(int(row[1]))

plt.hist(szkody, bins=50)
plt.show()

print ("Srednia wielkosc szkod:", round(sc.mean(szkody))) 


# In[ ]:


# wielkość szkód ma rozkład log-normalny:
szkody_ln = sc.log(szkody)

# ... czy faktycznie? test K-S
test3 = kstest(szkody_ln, sc.stats.norm.cdf, 
               args = (sc.mean(szkody_ln), sc.std(szkody_ln)))
if test3[1] > 0.05:
    print ("p-value wyniosło:", round(test3[1], 4), 
           "- brak więc podstaw do odrzucenia hipotezy " +
           "o log-normalności rozkładu zmiennej")
else:
    print ("należy odrzucic hipotezę zerowa")


# In[ ]:


# parametry wielkości szkód potrzebne do symulacji:
SR_SZKODA_LN = sc.mean(szkody_ln)
STD_SZKODA_LN = sc.std(szkody_ln)

# Przesunalem nizej, zeby miec zdefiniowane zmienne juz ~PS
fig = plt.figure(figsize = (10,5))
plt.hist(szkody_ln, bins=50, density = True, color = 'orange')
x_norm = np.linspace(min(szkody_ln), max(szkody_ln), 100)
y_norm = sc.stats.norm.pdf(x_norm, SR_SZKODA_LN, STD_SZKODA_LN)
plt.plot(x_norm, y_norm)
plt.legend(['Dopasowany r. normalny', 'Histogram rozkładu empirycznego'])
plt.ylabel('Gęstość masy prawdopodobieństwa')
fig.savefig(path+r'\rozklad_wys_szkody.pdf')

# ## 2. Model symulacji - podstawowy

# In[]:

def model (liczba_klientow, srednia_liczba_szkod, 
           sr_szkoda_ln, std_szkoda_ln, horyzont, 
           nadwyzka, skladka, seed):
    
    # Common Random Numbers aby moc porownac rozne scenariusze:
    sc.random.seed(seed) 

    # definiujemy daty umów klientow w symulacji:
    daty_umow = [sc.random.randint(0, 364) for i in range(liczba_klientow)]
    kalendarz_wplat = [0]*365
    for dataUmowy in daty_umow:
        kalendarz_wplat[dataUmowy] += 1

    # liczymy liczbe szkód przypadających na jednego klienta:
    liczba_szkod_klienta = sc.random.poisson(srednia_liczba_szkod, 
                                             liczba_klientow)
    
    # i ustalamy daty wyplaty dla wszystkich polis:
    kalendarz_wyplat = [0]*(365*horyzont) 
    for k in range(liczba_klientow):
        for s in range(liczba_szkod_klienta[k]):
            data_wyplaty = daty_umow[k] + sc.random.randint(0, 364)
            kalendarz_wyplat[data_wyplaty] += 1
    
    
    # analiza wyników firmy dla danego horyzontu czasowego i danych szkód:
    for dzien in range(365*horyzont):
        if dzien <= 364:
            nadwyzka += kalendarz_wplat[dzien] * skladka
        liczba_wyplat = kalendarz_wyplat[dzien]
        odszkodowania = 0 
        if liczba_wyplat > 0:
            odszkodowania = sum(sc.exp(sc.random.normal(sr_szkoda_ln, 
                                                        std_szkoda_ln, 
                                                        liczba_wyplat)))
        if nadwyzka < odszkodowania:
            return nadwyzka - odszkodowania
        else:
            pass
        nadwyzka -= odszkodowania
    return nadwyzka


# ### ... i funkcja, która go wywoła *n* razy

# In[ ]:

def wywolanie(nadwyzka, skladka, liczba_powtorzen, 
              liczba_klientow, srednia_liczba_szkod , 
              sr_szkoda_ln, std_szkoda_ln, horyzont):
    wynik = []
    bankructwo = 0
    wynik_dodatni = []
    for seed in range(liczba_powtorzen):
        wynik.append(model(liczba_klientow, srednia_liczba_szkod , 
                           sr_szkoda_ln, std_szkoda_ln, horyzont, 
                           nadwyzka, skladka, seed))
        if wynik[seed] < 0:
            bankructwo += 1
        if wynik[seed] > 0:
            wynik_dodatni.append(wynik[seed])
    sredni_wynik = sc.mean(wynik_dodatni)
    odch_stand_wynik = sc.std(wynik_dodatni)
    prawd_bankr = bankructwo / liczba_powtorzen
    return [bankructwo, prawd_bankr, sredni_wynik, odch_stand_wynik]


# ## 3. Symulacja

# In[ ]:


# zmienne i parametry w modelu:
sr_wynik = [] # średni wynik finansowy firmy
odch_stand_wynik = [] # Odch. stand. sredniego wyniku
wysokosc_skladki = []
wysokosc_nadwyzki = []
prawd_bankr = []
liczba_ruin = [] 

LICZBA_POWTORZEN = 1000
LICZBA_KLIENTOW = 100
HORYZONT = 2 # dlugość obowiązywania umowy - zakładamy 2 lata


for nadwyzka in range(10000, 30000, 10000):
    for skladka in range(500, 2000, 100):
        wartosc_f_xy = wywolanie(nadwyzka, skladka, 
                                 LICZBA_POWTORZEN, LICZBA_KLIENTOW, 
                                 SREDNIA_LICZBA_SZKOD , SR_SZKODA_LN, 
                                 STD_SZKODA_LN, HORYZONT)
        wysokosc_skladki.append(skladka)
        wysokosc_nadwyzki.append(nadwyzka)
        liczba_ruin.append(wartosc_f_xy[0])
        prawd_bankr.append(wartosc_f_xy[1])
        sr_wynik.append(wartosc_f_xy[2])
        odch_stand_wynik.append(wartosc_f_xy[3])
        print("Nadwyzka: ", nadwyzka, "Skladka: ", skladka, 
              "Liczba ruin: ", wartosc_f_xy[0], "Sredni wynik: ",
              round(wartosc_f_xy[2]), "Prawd_bankr: ", wartosc_f_xy[1])

dane = DataFrame.from_dict({'nadwyzka': wysokosc_nadwyzki,
                            'skladka': wysokosc_skladki,
                            'liczba_ruin': liczba_ruin,
                            'prawd_bankr': prawd_bankr,
                            'sr_wynik': sr_wynik,
                            'odch_stand_wynik': odch_stand_wynik},
        orient = 'columns')

dane.set_index(['nadwyzka', 'skladka'], inplace = True)

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[(10000,slice(None)),'prawd_bankr'].reset_index(). \
    plot(x = 'skladka', y = 'prawd_bankr', legend = False, ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
fig.savefig(path+r'\p_bankructwa.pdf')

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[:,'prawd_bankr'].reset_index().groupby('nadwyzka'). \
    plot(x = 'skladka', y = 'prawd_bankr', ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
ax.legend(labels = ['10 000', '20 000'], title = 'Nadwyżka [PLN]')
ax.axhline(y = 0.01, color = 'k' , linestyle = '--')
fig.savefig(path+r'\p_bankructwa_porownanie.pdf')


# Ponizej stary wykres p-stwa bankructwa (dla pojedynczej wys. nadwyzki)
#fig = plt.figure(figsize = (10,5))
#plt.plot(dane['wysokosc_skladki'], prawd_bankr)
#plt.ylabel('Prawdopodobienstwo bankructwa')
#plt.xlabel('Wysokość składki [PLN]')
#fig.savefig(path+'\p_bankructwa.pdf')



# %%
# Wykres sredniej koncowej nadwyzki dodatniej

plotdata = dane.loc[(10000, slice(None)),['sr_wynik', 'odch_stand_wynik']]
plotdata['upper'] = plotdata.sr_wynik + plotdata.odch_stand_wynik
plotdata['lower'] = plotdata.sr_wynik - plotdata.odch_stand_wynik

fig, ax = plt.subplots(figsize = (10,5))
plotdata.reset_index().plot(x = 'skladka', y = 'upper', legend = False,
                    ax = ax, kind = 'area', color = 'blue', alpha = 0.2)
plotdata.reset_index().plot(x = 'skladka', y = 'lower', legend = False,
                    ax = ax, kind = 'area', color = 'white')
plotdata.reset_index().plot(x = 'skladka', y = ['sr_wynik', 'upper', 'lower'],
                    legend = False, ax = ax, style = ['b', '--k', '--k'])
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Średnia nadwyżka końcowa [PLN]')
fig.savefig(path+r'\nadwyzka.pdf')


# Modele symulacji - analiza wrażliwosci

# In[ ]:


def model1 (liczba_klientow, srednia_liczba_szkod1, 
           sr_szkoda_ln, std_szkoda_ln, horyzont, 
           nadwyzka, skladka, seed):
    
    # Common Random Numbers aby moc porownac rozne scenariusze:
    sc.random.seed(seed) 

    # definiujemy daty umów klientow w symulacji:
    daty_umow = [sc.random.randint(0, 364) for i in range(liczba_klientow)]
    kalendarz_wplat = [0]*365
    for dataUmowy in daty_umow:
        kalendarz_wplat[dataUmowy] += 1

    # liczymy liczbe szkód przypadających na jednego klienta:
    liczba_szkod_klienta = sc.random.poisson(srednia_liczba_szkod1, 
                                             liczba_klientow)
    
    # i ustalamy daty wyplaty dla wszystkich polis:
    kalendarz_wyplat = [0]*(365*horyzont) 
    for k in range(liczba_klientow):
        for s in range(liczba_szkod_klienta[k]):
            data_wyplaty = daty_umow[k] + sc.random.randint(0, 364)
            kalendarz_wyplat[data_wyplaty] += 1
    
    
    # analiza wyników firmy dla danego horyzontu czasowego i danych szkód:
    for dzien in range(365*horyzont):
        if dzien <= 364:
            nadwyzka += kalendarz_wplat[dzien] * skladka
        liczba_wyplat = kalendarz_wyplat[dzien]
        odszkodowania = 0 
        if liczba_wyplat > 0:
            odszkodowania = sum(sc.exp(sc.random.normal(sr_szkoda_ln, 
                                                        std_szkoda_ln, 
                                                        liczba_wyplat)))
        if nadwyzka < odszkodowania:
            return nadwyzka - odszkodowania
        else:
            pass
        nadwyzka -= odszkodowania
    return nadwyzka


# ### ... i funkcja, która go wywoła *n* razy

# In[ ]:


def wywolanie(nadwyzka, skladka, liczba_powtorzen, 
              liczba_klientow, srednia_liczba_szkod1 , 
              sr_szkoda_ln, std_szkoda_ln, horyzont):
    wynik = []
    bankructwo = 0
    wynik_dodatni = []
    for seed in range(liczba_powtorzen):
        wynik.append(model1(liczba_klientow, srednia_liczba_szkod1 , 
                           sr_szkoda_ln, std_szkoda_ln, horyzont, 
                           nadwyzka, skladka, seed))
        if wynik[seed] < 0:
            bankructwo += 1
        if wynik[seed] > 0:
            wynik_dodatni.append(wynik[seed])
    sredni_wynik = sc.mean(wynik_dodatni)
    odch_stand_wynik = sc.std(wynik_dodatni)
    prawd_bankr = bankructwo / liczba_powtorzen
    return [bankructwo, prawd_bankr, sredni_wynik, odch_stand_wynik]


# ## 3. Symulacja

# In[ ]:


# zmienne i parametry w modelu:
sr_wynik = [] # średni wynik finansowy firmy
odch_stand_wynik = [] # Odch. stand. sredniego wyniku
wysokosc_skladki = []
wysokosc_nadwyzki = []
prawd_bankr = []
liczba_ruin = [] 

LICZBA_POWTORZEN = 1000
LICZBA_KLIENTOW = 100
HORYZONT = 2 # dlugość obowiązywania umowy - zakładamy 2 lata


for nadwyzka in range(10000, 30000, 10000):
    for skladka in range(500, 2000, 100):
        wartosc_f_xy = wywolanie(nadwyzka, skladka, 
                                 LICZBA_POWTORZEN, LICZBA_KLIENTOW, 
                                 SREDNIA_LICZBA_SZKOD1 , SR_SZKODA_LN, 
                                 STD_SZKODA_LN, HORYZONT)
        wysokosc_skladki.append(skladka)
        wysokosc_nadwyzki.append(nadwyzka)
        liczba_ruin.append(wartosc_f_xy[0])
        prawd_bankr.append(wartosc_f_xy[1])
        sr_wynik.append(wartosc_f_xy[2])
        odch_stand_wynik.append(wartosc_f_xy[3])
        print("Nadwyzka: ", nadwyzka, "Skladka: ", skladka, 
              "Liczba ruin: ", wartosc_f_xy[0], "Sredni wynik: ",
              round(wartosc_f_xy[2]), "Prawd_bankr: ", wartosc_f_xy[1])

dane = DataFrame.from_dict({'nadwyzka': wysokosc_nadwyzki,
                            'skladka': wysokosc_skladki,
                            'liczba_ruin': liczba_ruin,
                            'prawd_bankr': prawd_bankr,
                            'sr_wynik': sr_wynik,
                            'odch_stand_wynik': odch_stand_wynik},
        orient = 'columns')

dane.set_index(['nadwyzka', 'skladka'], inplace = True)

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[(10000,slice(None)),'prawd_bankr'].reset_index(). \
    plot(x = 'skladka', y = 'prawd_bankr', legend = False, ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
fig.savefig(path+r'\p_bankructwa_1.05.pdf')

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[:,'prawd_bankr'].reset_index().groupby('nadwyzka'). \
    plot(x = 'skladka', y = 'prawd_bankr', ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
ax.legend(labels = ['10 000', '20 000'], title = 'Nadwyżka [PLN]')
ax.axhline(y = 0.01, color = 'k' , linestyle = '--')
fig.savefig(path+r'\p_bankructwa_porownanie_1.05.pdf')


# Ponizej stary wykres p-stwa bankructwa (dla pojedynczej wys. nadwyzki)
#fig = plt.figure(figsize = (10,5))
#plt.plot(dane['wysokosc_skladki'], prawd_bankr)
#plt.ylabel('Prawdopodobienstwo bankructwa')
#plt.xlabel('Wysokość składki [PLN]')
#fig.savefig(path+'\p_bankructwa.pdf')

# %%
# Wykres sredniej koncowej nadwyzki dodatniej

plotdata = dane.loc[(10000, slice(None)),['sr_wynik', 'odch_stand_wynik']]
plotdata['upper'] = plotdata.sr_wynik + plotdata.odch_stand_wynik
plotdata['lower'] = plotdata.sr_wynik - plotdata.odch_stand_wynik

fig, ax = plt.subplots(figsize = (10,5))
plotdata.reset_index().plot(x = 'skladka', y = 'upper', legend = False,
                    ax = ax, kind = 'area', color = 'blue', alpha = 0.2)
plotdata.reset_index().plot(x = 'skladka', y = 'lower', legend = False,
                    ax = ax, kind = 'area', color = 'white')
plotdata.reset_index().plot(x = 'skladka', y = ['sr_wynik', 'upper', 'lower'],
                    legend = False, ax = ax, style = ['b', '--k', '--k'])
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Średnia nadwyżka końcowa [PLN]')
fig.savefig(path+r'\nadwyzka_1.05.pdf')


# In[ ]:


def model2 (liczba_klientow, srednia_liczba_szkod2, 
           sr_szkoda_ln, std_szkoda_ln, horyzont, 
           nadwyzka, skladka, seed):
    
    # Common Random Numbers aby moc porownac rozne scenariusze:
    sc.random.seed(seed) 

    # definiujemy daty umów klientow w symulacji:
    daty_umow = [sc.random.randint(0, 364) for i in range(liczba_klientow)]
    kalendarz_wplat = [0]*365
    for dataUmowy in daty_umow:
        kalendarz_wplat[dataUmowy] += 1

    # liczymy liczbe szkód przypadających na jednego klienta:
    liczba_szkod_klienta = sc.random.poisson(srednia_liczba_szkod2, 
                                             liczba_klientow)
    
    # i ustalamy daty wyplaty dla wszystkich polis:
    kalendarz_wyplat = [0]*(365*horyzont) 
    for k in range(liczba_klientow):
        for s in range(liczba_szkod_klienta[k]):
            data_wyplaty = daty_umow[k] + sc.random.randint(0, 364)
            kalendarz_wyplat[data_wyplaty] += 1
    
    
    # analiza wyników firmy dla danego horyzontu czasowego i danych szkód:
    for dzien in range(365*horyzont):
        if dzien <= 364:
            nadwyzka += kalendarz_wplat[dzien] * skladka
        liczba_wyplat = kalendarz_wyplat[dzien]
        odszkodowania = 0 
        if liczba_wyplat > 0:
            odszkodowania = sum(sc.exp(sc.random.normal(sr_szkoda_ln, 
                                                        std_szkoda_ln, 
                                                        liczba_wyplat)))
        if nadwyzka < odszkodowania:
            return nadwyzka - odszkodowania
        else:
            pass
        nadwyzka -= odszkodowania
    return nadwyzka


# ### ... i funkcja, która go wywoła *n* razy

# In[ ]:


def wywolanie(nadwyzka, skladka, liczba_powtorzen, 
              liczba_klientow, srednia_liczba_szkod2 , 
              sr_szkoda_ln, std_szkoda_ln, horyzont):
    wynik = []
    bankructwo = 0
    wynik_dodatni = []
    for seed in range(liczba_powtorzen):
        wynik.append(model2(liczba_klientow, srednia_liczba_szkod2 , 
                           sr_szkoda_ln, std_szkoda_ln, horyzont, 
                           nadwyzka, skladka, seed))
        if wynik[seed] < 0:
            bankructwo += 1
        if wynik[seed] > 0:
            wynik_dodatni.append(wynik[seed])
    sredni_wynik = sc.mean(wynik_dodatni)
    odch_stand_wynik = sc.std(wynik_dodatni)
    prawd_bankr = bankructwo / liczba_powtorzen
    return [bankructwo, prawd_bankr, sredni_wynik, odch_stand_wynik]


# ## 3. Symulacja

# In[ ]:


# zmienne i parametry w modelu:
sr_wynik = [] # średni wynik finansowy firmy
odch_stand_wynik = [] # Odch. stand. sredniego wyniku
wysokosc_skladki = []
wysokosc_nadwyzki = []
prawd_bankr = []
liczba_ruin = [] 

LICZBA_POWTORZEN = 1000
LICZBA_KLIENTOW = 100
HORYZONT = 2 # dlugość obowiązywania umowy - zakładamy 2 lata


for nadwyzka in range(10000, 30000, 10000):
    for skladka in range(500, 2000, 100):
        wartosc_f_xy = wywolanie(nadwyzka, skladka, 
                                 LICZBA_POWTORZEN, LICZBA_KLIENTOW, 
                                 SREDNIA_LICZBA_SZKOD2 , SR_SZKODA_LN, 
                                 STD_SZKODA_LN, HORYZONT)
        wysokosc_skladki.append(skladka)
        wysokosc_nadwyzki.append(nadwyzka)
        liczba_ruin.append(wartosc_f_xy[0])
        prawd_bankr.append(wartosc_f_xy[1])
        sr_wynik.append(wartosc_f_xy[2])
        odch_stand_wynik.append(wartosc_f_xy[3])
        print("Nadwyzka: ", nadwyzka, "Skladka: ", skladka, 
              "Liczba ruin: ", wartosc_f_xy[0], "Sredni wynik: ",
              round(wartosc_f_xy[2]), "Prawd_bankr: ", wartosc_f_xy[1])

dane = DataFrame.from_dict({'nadwyzka': wysokosc_nadwyzki,
                            'skladka': wysokosc_skladki,
                            'liczba_ruin': liczba_ruin,
                            'prawd_bankr': prawd_bankr,
                            'sr_wynik': sr_wynik,
                            'odch_stand_wynik': odch_stand_wynik},
        orient = 'columns')

dane.set_index(['nadwyzka', 'skladka'], inplace = True)

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[(10000,slice(None)),'prawd_bankr'].reset_index(). \
    plot(x = 'skladka', y = 'prawd_bankr', legend = False, ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
fig.savefig(path+r'\p_bankructwa_0.95.pdf')

fig, ax = plt.subplots(figsize = (10,5))
dane.loc[:,'prawd_bankr'].reset_index().groupby('nadwyzka'). \
    plot(x = 'skladka', y = 'prawd_bankr', ax = ax)
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Prawdopodobieństwo bankructwa')
ax.legend(labels = ['10 000', '20 000'], title = 'Nadwyżka [PLN]')
ax.axhline(y = 0.01, color = 'k' , linestyle = '--')
fig.savefig(path+r'\p_bankructwa_porownanie_0.95.pdf')


# %%
# Wykres sredniej koncowej nadwyzki dodatniej

plotdata = dane.loc[(10000, slice(None)),['sr_wynik', 'odch_stand_wynik']]
plotdata['upper'] = plotdata.sr_wynik + plotdata.odch_stand_wynik
plotdata['lower'] = plotdata.sr_wynik - plotdata.odch_stand_wynik

fig, ax = plt.subplots(figsize = (10,5))
plotdata.reset_index().plot(x = 'skladka', y = 'upper', legend = False,
                    ax = ax, kind = 'area', color = 'blue', alpha = 0.2)
plotdata.reset_index().plot(x = 'skladka', y = 'lower', legend = False,
                    ax = ax, kind = 'area', color = 'white')
plotdata.reset_index().plot(x = 'skladka', y = ['sr_wynik', 'upper', 'lower'],
                    legend = False, ax = ax, style = ['b', '--k', '--k'])
ax.set_xlabel('Wysokość składki [PLN]')
ax.set_ylabel('Średnia nadwyżka końcowa [PLN]')
fig.savefig(path+r'\nadwyzka_0.95.pdf')

# In[]:

#Inny kod
import numpy.random as rd

def symuluj_ubezpieczenia(l_klientow,nadwyzka,skladka,srednia_l_szkod,sr_szkoda_ln,std_szkoda_ln):
    daty_umow = rd.randint(0,365, l_klientow)
    kal_l_wplat = np.zeros(365+365+30, dtype="int")
    for dataUmowy in daty_umow:
        kal_l_wplat[dataUmowy] += 1
    l_szkod_k = rd.poisson(srednia_l_szkod,l_klientow)
    kal_l_wyplat = np.zeros(365+365+30, dtype="int") #365 to zapas
    for k in range(l_klientow):
        for s in range(l_szkod_k[k]):
            #dla kazdej szkody ustal date wyplaty
            data_wyp = daty_umow[k]+rd.randint(0,365)+rd.randint(15,30)
            kal_l_wyplat[data_wyp] += 1
    for dzien in range(len(kal_l_wyplat)):
        nadwyzka += kal_l_wplat[dzien]*skladka
        l_wyplat = kal_l_wyplat[dzien]
        odszkodowania = 0
        if l_wyplat>0:
            odszkodowania=np.sum(np.exp(rd.normal(sr_szkoda_ln,std_szkoda_ln,l_wyplat)))
        if (nadwyzka<odszkodowania):
            return (nadwyzka-odszkodowania, dzien)
        nadwyzka -= odszkodowania
    return (nadwyzka, dzien)

# In[]:
    
def run_symulacja(blok):
    l_szkod = { # liczba szkod : liczba polis
            0 : 3437,
            1 : 522,
            2 : 40,
            3 : 2,
            4 : 0,
            5 : 0
    }
    
    srednia_l_szkod = sum( [x*y for x,y in l_szkod.items()] )*1./sum(l_szkod.values())
    sr_szkoda_ln = 7.9953648143576634
    std_szkoda_ln = 0.9644771368064744
    for skladka in range(500+blok*100,500+(blok+1)*100,25):
            rd.seed(0)
            wyniki=[symuluj_ubezpieczenia(10000,10000,\
                                          skladka,srednia_l_szkod,sr_szkoda_ln,std_szkoda_ln) \
                for i in range(100)]
            srednia = np.mean([x[0] for x in wyniki if x[0] >= 0])
            liczba_ruin = np.sum([1 for x in wyniki if x[0] < 0])
            sredni_dzien_ruiny = np.mean([x[1] for x in wyniki if x[0] < 0])
            return([skladka,srednia,liczba_ruin,sredni_dzien_ruiny])

# In[]:
sredni_dzien_ruiny = []
srednia = []
wysokosc_skladki = []   
liczba_ruin = []  
       
for nadwyzka in range(10000, 20000, 10000):
        wartosc_g_xy = run_symulacja(1)
        wysokosc_skladki.append(wartosc_g_xy[0])
        liczba_ruin.append(wartosc_g_xy[2])
        sredni_dzien_ruiny.append(wartosc_g_xy[3])
        srednia.append(wartosc_g_xy[1])
        print("Nadwyzka: ", nadwyzka, "Skladka: ", wartosc_g_xy[0], 
              "Liczba ruin: ", wartosc_g_xy[2], "Sredni wynik: ",
              round(wartosc_g_xy[1]), "Sredni dzien ruiny: ", wartosc_g_xy[3])
