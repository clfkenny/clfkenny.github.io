

```python
from IPython.core.display import HTML
HTML("""
<style>
    table.dataframe thead th:first-child {
        display: none;
    }
    table.dataframe tbody th {
        display: none;
    }
</style>
""")
```




    <IPython.core.display.HTML object>



# Obtaining the Data

Before we can analyze data, we first need to obtain something to analyze. There are a lot of great websites that have the information we need, but for this example we'll be going with [Pokemon Database](https://www.pokemondb.net/pokedex/all). Instead of manually iterating through every single page writing down information about their battle stats, a much more efficient way would be to program a webscraper to automate the data collection. We'll be using Python's `resquest` module to send requests to the website and `BeautifulSoup` to parse the data. Let's begin. 


```python
# import modules for web scraping
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm # add a progress bar for loops
```


```python
main_page = requests.get('https://pokemondb.net/pokedex/all')
soup = BeautifulSoup(main_page.content, 'html.parser')

poke_html_list = soup.select('a.ent-name')
poke_list = []
for poke in poke_html_list:
    if poke['href'] not in poke_list: poke_list.append(poke['href'])
```


```python
pokemon_list = []
base_stats = []
type_ = []
evo_stage_list = []

for pokemon in tqdm(poke_list):
    page2 = requests.get('https://pokemondb.net' + pokemon)
    soup2 = BeautifulSoup(page2.content, 'html.parser')
    
    stats = soup2.select('table.vitals-table tbody td.cell-num') #selecting all the numbers
    stat_numbers = []
    for index, i in enumerate(stats):
        if (index)%3 ==0: stat_numbers.append(i.text) #every 3 is one of the main stats
    base_stats.append(stat_numbers)
    
    
    types = soup2.select('table.vitals-table tbody tr a.type-icon')
    ind_type = []
    for i in types:
        ind_type.append(i.text)
    type_.append(list(set(ind_type)))

    evo_list = soup2.select('div.infocard a.ent-name')
    evo_list = list(map(str, evo_list))
    evo_stage = [stage for stage, item in enumerate(evo_list) if pokemon+'"' in item]
    # if the list is empty, then that means there is no other evolutions, so just assign it '1'
    try:
        evo_stage_list.append(evo_stage[0] + 1)
    except:
        evo_stage_list.append(1)
```


```python
hp = []
att = []
defs = []
spatt = []
spdef = []
spe = []

for i in base_stats:
    hp.append(i[0])
    att.append(i[1])
    defs.append(i[2])
    spatt.append(i[3])
    spdef.append(i[4])
    spe.append(i[5])
    
first_type = []
for i in type_:
    first_type.append(i[0])

import pandas as pd
dataf = pd.DataFrame({'pokemon': [i.replace('/pokedex/', '') for i in poke_list],
                      'hp': hp,
                      'att': att,
                      'defs': defs,
                      'spatt': spatt,
                      'spdef': spdef,
                      'spe': spe,
                      'type': first_type,
                      'types': type_,
                     'evo_stage': evo_stage_list})

dataf = dataf[['pokemon']+list(dataf.drop('pokemon', axis=1).columns)] # moving pokemon column to the front
dataf.to_csv('original_data.csv', index=False)
```

This program iterates through all of the individual pokemon's unique page and collects their HP, Attack, Defense, Sp. Attack, Sp. Def, Speed, Types, and their Evolution Stage. Once the information is collected, the data is passed into a pandas `DataFrame` object so we get an organized and informative dataframe! This is what the first 10 rows look like:


```python
dataf.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokemon</th>
      <th>hp</th>
      <th>att</th>
      <th>defs</th>
      <th>spatt</th>
      <th>spdef</th>
      <th>spe</th>
      <th>type</th>
      <th>types</th>
      <th>evo_stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bulbasaur</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>Grass</td>
      <td>[Grass, Poison]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ivysaur</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>Grass</td>
      <td>[Grass, Poison]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>venusaur</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>Grass</td>
      <td>[Grass, Poison]</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>charmander</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>Fire</td>
      <td>[Fire]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmeleon</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>Fire</td>
      <td>[Fire]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charizard</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>109</td>
      <td>85</td>
      <td>100</td>
      <td>Fire</td>
      <td>[Fire, Flying, Dragon]</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>squirtle</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>50</td>
      <td>64</td>
      <td>43</td>
      <td>Water</td>
      <td>[Water]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wartortle</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>58</td>
      <td>Water</td>
      <td>[Water]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>blastoise</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>85</td>
      <td>105</td>
      <td>78</td>
      <td>Water</td>
      <td>[Water]</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>caterpie</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>20</td>
      <td>20</td>
      <td>45</td>
      <td>Bug</td>
      <td>[Bug]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The data is formatted such that the columns represent the different stats, types, and evolution stage while each row represents a different pokemon. This convenient `DataFrame` structure will allow us easily perform our analyses. Now that the computer has done all the hard work of collecting the data, let's dive in!

# Exploratory Data Analysis


```python
# importing our data exploration modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams.update({'font.size': 20}) # to increase matplotlib's font size
sns.set_style("darkgrid") # set plot styling
```

We'll be using `pandas` for all of our data manipulation needs. For visualizations, we'll be using `seaborn`, which is a statisical data visualization library that uses `matplotlib`. To pass the data into `seaborn`, we first need to "melt" the data so that it is in *long* format rather than *wide*. This means that there will be a single column called `stats` rather than having each stat as a separate column.


```python
dataf = pd.read_csv('original_data.csv')
dataf.loc[dataf.evo_stage>3, 'evo_stage'] = 3
dataf.pokemon = dataf.pokemon.str.lower()
dataf_stats = dataf.drop(['types'], axis=1)

dataf_long_stats = pd.melt(dataf_stats, id_vars = ['pokemon','type','evo_stage'], var_name = 'stats')
dataf_long_stats.head(10)
print('The shape of the original dataframe is: ', dataf.shape)
print('The shape of the melted dataframe is: ', dataf_long_stats.shape)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokemon</th>
      <th>type</th>
      <th>evo_stage</th>
      <th>stats</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>1</td>
      <td>hp</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>2</td>
      <td>hp</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>3</td>
      <td>hp</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>1</td>
      <td>hp</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>2</td>
      <td>hp</td>
      <td>58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charizard</td>
      <td>Fire</td>
      <td>3</td>
      <td>hp</td>
      <td>78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>squirtle</td>
      <td>Water</td>
      <td>1</td>
      <td>hp</td>
      <td>44</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wartortle</td>
      <td>Water</td>
      <td>2</td>
      <td>hp</td>
      <td>59</td>
    </tr>
    <tr>
      <th>8</th>
      <td>blastoise</td>
      <td>Water</td>
      <td>3</td>
      <td>hp</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9</th>
      <td>caterpie</td>
      <td>Bug</td>
      <td>1</td>
      <td>hp</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



    The shape of the original dataframe is:  (807, 10)
    The shape of the melted dataframe is:  (4842, 5)


We can see that the new dataframe is 807*6=4842. Instead of having a "wider" dataframe, the dataframe is now "longer," which makes it easier to perform aggregate computations and visualizations. First, let's see the average value of each stat using `pivot_table`.


```python
df_agg = dataf_long_stats.groupby('stats')
df_agg.value.agg([np.mean, np.std])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>stats</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>att</th>
      <td>76.210657</td>
      <td>29.644590</td>
    </tr>
    <tr>
      <th>defs</th>
      <td>71.602230</td>
      <td>29.611741</td>
    </tr>
    <tr>
      <th>hp</th>
      <td>68.748451</td>
      <td>26.032808</td>
    </tr>
    <tr>
      <th>spatt</th>
      <td>69.610905</td>
      <td>29.567768</td>
    </tr>
    <tr>
      <th>spdef</th>
      <td>69.889715</td>
      <td>27.155402</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>65.830235</td>
      <td>27.736838</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's see the distributions of each stat with boxplots.


```python
plt.figure(figsize=(8,6))
sns.boxplot(x='stats', y='value', data = dataf_long_stats)
plt.title('Boxplot of Stats')
```




    <Figure size 576x432 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x7fe2bfb39ac8>






    Text(0.5,1,'Boxplot of Stats')




![png](output_17_3.png)


Doesn't seem like there's anything *toooo* interesting here. The medians for each stat appears to be around 70-80. There does seem to be an abnormally large amount of outliers for HP. Let's take a look at how evolution stage affects stats using a swarmplot. Since the visual will be very messy with all the pokemon, I decided to take a look at the first 200.


```python
# script to extract first 200 pokemon
indexes = []
count = 0
for i in range(dataf_long_stats.shape[0]):
    if count <= 200:
        indexes.append(i)               
    count += 1               
    if i % 807 == 0:
        count = 0    

plt.figure(figsize=(10,10))
sns.swarmplot(x='stats', y= 'value', hue = 'evo_stage', 
              data = dataf_long_stats.iloc[indexes], palette = 'muted' )
plt.title('Swarmplot of Stats with Evolution')
```




    <Figure size 720x720 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x7fe2c53be518>






    Text(0.5,1,'Swarmplot of Stats with Evolution')




![png](output_19_3.png)


This graph tells us a few things that we probably already knew. First of all, we can see that the first stage comprises most of the pokemon and tends to have the lowest stats. Next, we see that the third stage are the fewest but they tend to be higher up on the graph. The second stage is intermediate in both count and stat values. 


```python
plt.figure(figsize=(15,13))
sns.swarmplot(x='stats', y = 'value', hue = 'type', data = dataf_long_stats)
plt.legend(loc = 2, bbox_to_anchor=(1,1))
```




    <Figure size 1080x936 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x7fe2e5b059b0>






    <matplotlib.legend.Legend at 0x7fe2e5688e10>




![png](output_21_3.png)


Let's see what kind of role type plays on stats. First let's take a look at the numbers by aggregating the data with a pivot table.


```python
df_agg = pd.pivot_table(dataf_long_stats.drop('evo_stage',axis=1), 
                        index='type', columns = 'stats', values='value')
df_agg
sns.heatmap(df_agg)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stats</th>
      <th>att</th>
      <th>defs</th>
      <th>hp</th>
      <th>spatt</th>
      <th>spdef</th>
      <th>spe</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>57.575000</td>
      <td>66.625000</td>
      <td>50.275000</td>
      <td>41.575000</td>
      <td>57.750000</td>
      <td>50.150000</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>84.800000</td>
      <td>61.200000</td>
      <td>62.000000</td>
      <td>84.800000</td>
      <td>68.700000</td>
      <td>78.200000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>93.678571</td>
      <td>78.750000</td>
      <td>79.714286</td>
      <td>82.964286</td>
      <td>80.392857</td>
      <td>70.357143</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>69.454545</td>
      <td>56.515152</td>
      <td>63.181818</td>
      <td>79.424242</td>
      <td>65.606061</td>
      <td>82.242424</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>63.333333</td>
      <td>71.361111</td>
      <td>64.833333</td>
      <td>79.000000</td>
      <td>87.500000</td>
      <td>63.361111</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>99.227273</td>
      <td>73.318182</td>
      <td>74.113636</td>
      <td>60.227273</td>
      <td>69.386364</td>
      <td>72.568182</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>80.500000</td>
      <td>65.515625</td>
      <td>68.328125</td>
      <td>87.578125</td>
      <td>70.218750</td>
      <td>75.171875</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>76.157895</td>
      <td>69.070175</td>
      <td>73.087719</td>
      <td>80.210526</td>
      <td>74.789474</td>
      <td>83.982456</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>72.142857</td>
      <td>87.428571</td>
      <td>56.142857</td>
      <td>79.285714</td>
      <td>88.428571</td>
      <td>59.857143</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>73.052632</td>
      <td>71.294737</td>
      <td>64.968421</td>
      <td>71.094737</td>
      <td>70.747368</td>
      <td>57.305263</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>88.166667</td>
      <td>83.900000</td>
      <td>74.366667</td>
      <td>55.783333</td>
      <td>61.033333</td>
      <td>57.500000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>78.642857</td>
      <td>78.500000</td>
      <td>67.000000</td>
      <td>71.142857</td>
      <td>82.214286</td>
      <td>65.571429</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>73.588785</td>
      <td>58.457944</td>
      <td>76.093458</td>
      <td>56.906542</td>
      <td>63.028037</td>
      <td>69.177570</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>70.720000</td>
      <td>64.800000</td>
      <td>64.520000</td>
      <td>65.440000</td>
      <td>64.480000</td>
      <td>67.320000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>55.378378</td>
      <td>63.432432</td>
      <td>70.567568</td>
      <td>85.324324</td>
      <td>79.135135</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>87.629630</td>
      <td>94.888889</td>
      <td>70.925926</td>
      <td>61.185185</td>
      <td>67.333333</td>
      <td>50.481481</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>91.000000</td>
      <td>111.000000</td>
      <td>65.171429</td>
      <td>73.542857</td>
      <td>81.142857</td>
      <td>55.485714</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>70.543210</td>
      <td>70.518519</td>
      <td>67.629630</td>
      <td>70.802469</td>
      <td>65.604938</td>
      <td>63.407407</td>
    </tr>
  </tbody>
</table>
</div>






    <matplotlib.axes._subplots.AxesSubplot at 0x7fe2c41b10f0>




![png](output_23_2.png)


It's a bit hard to see any trends with just numbers... Let's see how these numbers play out with visuals. This will be a barplot that describes the distribution of means for each of the stats.


```python
plt.rcParams.update({'font.size': 13})
g = sns.factorplot(x='stats', y = 'value', data = dataf_long_stats,
                   col = 'type', col_wrap=5, kind = 'bar', size = 3)
```


![png](output_25_0.png)


A bit hard to see, but there are some interesting trends. For example, bug type pokemon seem to have lower overall. Steel and rock types have strikingly high defense but very low speed. 


```python
plt.rcParams.update({'font.size': 13})
g = sns.factorplot(x='stats', y = 'value', data = dataf_long_stats,
                   col = 'type', col_wrap=5, kind = 'point', size = 3)
```


![png](output_27_0.png)


A similar visual, except using points rather than bar. This allows us to much more easily see the distribution of stats per type.

# Obtaining the Main Color of each Pokemon


```python
import requests
import PIL
import io
import numpy as np
import cv2
```


```python
# Download all of the pokemon images

poke_col_list = []
for counter, pokemon in enumerate(poke_list):
    page2 = requests.get('https://pokemondb.net' + pokemon)
    soup2 = BeautifulSoup(page2.content, 'html.parser')

    img_link = soup2.select('div.col.desk-span-4.lap-span-6.figure img')[0]['src']
    #print(img_link)
    

    img = requests.get(img_link, stream=True)
    
    img_name = 'images' + pokemon.replace('/pokedex', '') + '.jpg' 
    with open(img_name, 'wb') as handler:
        handler.write(img.content)
        #print('Saving', pokemon)
        print(str(counter+1)+ '/'+ str(len(poke_list)) + ' saved.')    
```

I will be defining the "main color" of the pokemon as the average of it's color. Each image contains the pokemon and a white background. If I take just the average of the entire image, it will be too white because of the white background. To avoid this problem, I used a k-means function to compute 2 clusters. One cluster center will correspond to the average of the white background while the other cluster center will correspond to the average of the pokemon color.


```python
from os import listdir
poke_img_files = listdir('images')

poke_col_list = []
for poke_img in tqdm(poke_img_files):
    img = PIL.Image.open('images/'+ poke_img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))


    b,g,r = cv2.split(res2)
    frame_rgb = cv2.merge((r,g,b))
    img_name = 'new_images/' + poke_img
    pil_img = PIL.Image.fromarray(frame_rgb, "RGB")
    pil_img.save(img_name)
    #print('Saving', poke_img)
    
    # extracting the two colors
    two_colors = []
    big_list = []
    res3 = res2.tolist()
    for i in res3:
        big_list.extend(i)
    for i in big_list:
        in_list = i in two_colors
        if in_list == False: two_colors.append(i)
    # the color that is farther away from 250 (white) is probably the pokemon's dominant color

    main_color = [sorted(two_colors)[0]] # each array is (BGR) instead of (RGB)
    poke_col_list.append(main_color)
```


```python
poke_col_list_flat = []
for i in poke_col_list:
    for j in i:
        poke_col_list_flat.append(j)
b, g, r = zip(*poke_col_list_flat) # create new lists that correspond to blue, green, and red channel
color_df = pd.DataFrame({'r':r, 'g': g, 'b': b,
                         'pokemon': [item.replace('.jpg', '') for item in poke_img_files]})

# adding this column 'color' because plotly requires color to be in that format
color_df['color'] = 'rgb(' + color_df.r.map(str) + ', ' + color_df.g.map(str) + ', ' + color_df.b.map(str) + ')'

dataf = pd.merge(dataf, color_df, on = 'pokemon') # merging the color dataframe with the original dataframe by 'pokemon' column
dataf.shape
dataf.head()

dataf.to_csv('data_w_color.csv', index= False)
```




    (807, 14)






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokemon</th>
      <th>hp</th>
      <th>att</th>
      <th>defs</th>
      <th>spatt</th>
      <th>spdef</th>
      <th>spe</th>
      <th>type</th>
      <th>types</th>
      <th>evo_stage</th>
      <th>r</th>
      <th>g</th>
      <th>b</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bulbasaur</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>Grass</td>
      <td>['Grass', 'Poison']</td>
      <td>1</td>
      <td>98</td>
      <td>153</td>
      <td>119</td>
      <td>rgb(98, 153, 119)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ivysaur</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>Grass</td>
      <td>['Grass', 'Poison']</td>
      <td>2</td>
      <td>95</td>
      <td>133</td>
      <td>133</td>
      <td>rgb(95, 133, 133)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>venusaur</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>Grass</td>
      <td>['Grass', 'Poison']</td>
      <td>3</td>
      <td>98</td>
      <td>113</td>
      <td>107</td>
      <td>rgb(98, 113, 107)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>charmander</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>Fire</td>
      <td>['Fire']</td>
      <td>1</td>
      <td>188</td>
      <td>138</td>
      <td>97</td>
      <td>rgb(188, 138, 97)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmeleon</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>Fire</td>
      <td>['Fire']</td>
      <td>2</td>
      <td>191</td>
      <td>102</td>
      <td>84</td>
      <td>rgb(191, 102, 84)</td>
    </tr>
  </tbody>
</table>
</div>



Visualizing the the colors


```python
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
```


```python
dataf = pd.read_csv('data_w_color.csv')

trace1 = go.Scatter3d(
    x=dataf['r'],
    y=dataf['g'],
    z=dataf['b'],
    mode='markers',
    marker=dict(
        color=dataf['color'],
        size=5,
        line=dict(
            color= dataf['color'],
            width=0.5
        ),
        opacity=0.8
    ),
    text = dataf.pokemon
)

data = [trace1]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
        xaxis = dict(title = 'Red'),
        yaxis = dict(title = 'Green'),
        zaxis = dict(title = 'Blue')
    ),
    paper_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~kennylov/61.embed" height="525px" width="100%"></iframe>



Interesting results... but this doesn't tell us much besides the color distribution. What's more important would be to include type into visual. Now, keeping the same points on this graph, but changing each point's color to correspond to its type. This will allow us to see the relationship between color and type. First, we need to create a new color scheme that corresponds to the colors that we're more familiar with.


```python
color_type_dict = {}

# obtaining this color palette from bulbapedia.com
color_type_dict = {'Grass': 'rgb(120,200,80)',
                   'Fire': 'rgb(240,128,48)',
                   'Water': 'rgb(104,144,240)',
                   'Bug': 'rgb(168,184,32)',
                   'Normal': 'rgb(168,168,120)',
                   'Poison': 'rgb(160,64,160)',
                   'Electric': 'rgb(248,208,48)',
                   'Ground': 'rgb(224,192,104)',
                   'Fairy': 'rgb(238,153,172)',
                   'Fighting': 'rgb(192,48,40)',
                   'Psychic': 'rgb(248,88,136)',
                   'Rock': 'rgb(184,160,56)',
                   'Ghost': 'rgb(112,88,152)',
                   'Ice': 'rgb(152,216,216)',
                   'Dragon': 'rgb(112,56,248)',
                   'Flying': 'rgb(168,144,240)',
                   'Dark': 'rgb(112,88,72)',
                   'Steel': 'rgb(184,184,208)'
}    

# mapping these colors to each pokemon    
color_type = []
for i in range(0,len(dataf)):
    p_type = dataf.iloc[i].type
    color_item = color_type_dict[p_type]
    col = color_item
    color_type.append(col)
    
dataf['color_type'] = color_type
```


```python
trace2 = []

for col_type in np.unique(color_type):
    dataf_type_subset = dataf.loc[dataf.color_type == col_type]
    trace2.append(
        go.Scatter3d(
        x=dataf_type_subset['r'],
        y=dataf_type_subset['b'],
        z=dataf_type_subset['g'],
        mode='markers',
        marker=dict(
            color=dataf_type_subset['color_type'],
            size=5,
            line=dict(
                color= dataf_type_subset['color_type'],
                width=0.5
            ),
            opacity=0.8
        ),
        text = dataf_type_subset['type'],
        name = dataf_type_subset.type.iloc[1]
        )
    )

    
data2 = [trace2]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
        xaxis = dict(title = 'Red'),
        yaxis = dict(title = 'Blue'),
        zaxis = dict(title = 'Green')
    ),
    paper_bgcolor='rgba(0,0,0,0)'
)

fig2 = go.Figure(data=trace2, layout=layout)
py.iplot(fig2, filename='simple-3d-scatter2')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~kennylov/63.embed" height="525px" width="100%"></iframe>



The pokemon are in the same space as the previous graph. The difference here is that the color of the points changed to represent the pokemon's type. Since there are so many different types in such a small plot, it's difficult to see clear clusters. However, there are distinct clusters; for example, you can see that fire types and water types are on opposite ends. Just out of curiousity I'll try different machine learning methods to see how useful of a feature color is.

# Building a Predictive Model

Just a reminder, we'll be trying to predict a pokemon's type by using their base stats, evolution stage, and color.
The first thing that should be done is to establish a benchmark to beat. This benchmark will be the accuracy of a "dumb" model that predicts only the most frequent type. Since the most frequent type is water, this "dumb" model will have an accuracy of about 116/807 $\approx$ 14%

Next, we'll split up the data into a training set (80%) and a validation set (20%) to see how well the model generalizes to data it hasn't seen before.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

# xgboost causing warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
dataf = pd.read_csv('data_w_color.csv')

# first drop all unnecessary variables
dataf_clean = dataf.drop(['types', 'color', 'pokemon'], axis=1)
X = dataf_clean.drop('type', axis=1) # feature matrix
y = dataf_clean.type # target vector
```


```python
# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123, stratify = y)
print('The shape of X_train is ', X_train.shape)
print('The shape of y_train is ', y_train.shape)
print('The shape of X_test is ', X_test.shape)
print('The shape of y_test is ', y_test.shape)
```

    The shape of X_train is  (645, 10)
    The shape of y_train is  (645,)
    The shape of X_test is  (162, 10)
    The shape of y_test is  (162,)


Let's first try out a few classical machine learning algorithms to see which ones perform well by looking at their cross validation scores


```python
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=10)))
models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=200,
                                                                criterion='gini',
                                                                    max_features=3)))
models.append(('XGBoost', XGBClassifier(learning_rate=0.05, n_estimators=100,
                                       max_depth= 5)))

results = []
names = []
for name,model in models:
    result = cross_val_score(model, X_train, y_train, cv=5)
    names.append(name)
    results.append(result)

mean_results = []
sd_results = []
for model in results:
    mean_results.append(model.mean())
    sd_results.append(model.std())

results_df = pd.DataFrame({'Model': names,
              'Mean Classification Rate': mean_results,
              'Standard Deviation': sd_results
             })
results_df.sort_values(by = 'Mean Classification Rate', ascending=False) # returning sorted dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Mean Classification Rate</th>
      <th>Standard Deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>0.307604</td>
      <td>0.038762</td>
    </tr>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.289793</td>
      <td>0.027475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNeighborsClassifier</td>
      <td>0.261504</td>
      <td>0.027064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBoost</td>
      <td>0.258269</td>
      <td>0.030368</td>
    </tr>
  </tbody>
</table>
</div>



Seems that a simple logistic regression has the highest cross-validation score. Let's see how they do on the held-out validation set.


```python
results = []
names = []
for name, model in models:
    model_fit = model.fit(X_train, y_train)
    pred = model_fit.predict(X_test)
    result = accuracy_score(pred, y_test)
    results.append(result)
    names.append(name)

results_df = pd.DataFrame({
    'Model': names,
    'Classification Rate': results
})

results_df.sort_values(by = 'Classification Rate', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Classification Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.339506</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNeighborsClassifier</td>
      <td>0.327160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>0.314815</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBoost</td>
      <td>0.302469</td>
    </tr>
  </tbody>
</table>
</div>



Now let's try out neural networks, which I think are the coolest things ever.


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler

import keras.backend as K
```

    Using TensorFlow backend.



```python
K.clear_session()

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


def build_nnet():
    nnet = Sequential()
    nnet.add(Dense(256, kernel_initializer= 'normal', input_shape = (10,), activation = 'relu'))
    nnet.add(Dense(256, kernel_initializer= 'normal',activation = 'relu'))
    nnet.add(Dense(256, kernel_initializer= 'normal',activation = 'relu'))

    nnet.add(Dense(18, activation = 'softmax'))
    nnet.compile(Adam(lr=0.001), 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return nnet

y_train_cat = pd.get_dummies(y_train)
y_test_cat = pd.get_dummies(y_test)
nnet = KerasClassifier(build_fn = build_nnet, epochs = 100, verbose =0)
```


```python
cv_score = cross_val_score(nnet, X_train, y_train_cat)
print('The cross-validation scores for this neural network are: ', cv_score)
print('The average cross-validation score for this neural network is: ', cv_score.mean())
```

    The cross-validation scores for this neural network are:  [0.3255814  0.31627907 0.30697674]
    The average cross-validation score for this neural network is:  0.31627906976744186



```python
nnet.fit(X_train, y_train_cat, epochs = 100, verbose=0, batch_size=32)

pred = nnet.predict(X_test)
#pred
accuracy_score(np.asarray(y_test_cat).argmax(axis=1),pred)
```




    <keras.callbacks.History at 0x7f07a5b31940>






    0.345679012345679



# Conclusion
