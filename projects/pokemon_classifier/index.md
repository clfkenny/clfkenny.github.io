---
layout: projects
title: Pokemon Type Classifier
author: Kenny Lov
date: 4/26/2018
---

# __Pokemon Type Classifier__
*Kenny Lov*

*4/26/2018*


## Introduction

<style> 
  img#pokemon-logo {
  position: relative;
  bottom: 20px;
}

img#pokemon-logo {
    -webkit-animation: fadein 3s; /* Safari, Chrome and Opera > 12.1 */
       -moz-animation: fadein 3s; /* Firefox < 16 */
        -ms-animation: fadein 3s; /* Internet Explorer */
         -o-animation: fadein 3s; /* Opera < 12.1 */
            animation: fadein 3s;
}
@-moz-keyframes fadein {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@-webkit-keyframes fadein {
    from { opacity: 0; }
    to   { opacity: 1; }
}

</style>

<div id = "poke_logo" style = "text-align: center"> 

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script> 
<script src="//code.jquery.com/jquery-1.12.4.js"></script> 
<script src="//code.jquery.com/ui/1.12.1/jquery-ui.js"></script>



<img id = "pokemon-logo" src="pokemon.png">
</div>

Looking back to my childhood, I have very fond memories of Pokemon, whether it be from waiting for new episodes to air or from the many hours I had played the games. I noticed, while playing the games, that certain types of Pokemon had better base stats than others. For example, I noticed that rock and steel types had especially high defense, fighting and steel types had especially high attack, and dragon types had high values over all base stats. I think it would be interesting to see if there's any correlation between base stats and types, and if so, would it be possible to predict a pokemon's type given their base stats? 

<br>
<br>
*The goal will be to see if we can correctly predict a pokemon's type based on their base stats.*

## Obtaining the Data


```python
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm # add a progress bar
```


```python
main_page = requests.get('https://pokemondb.net/pokedex/all')
soup = BeautifulSoup(main_page.content, 'html.parser')
```


```python
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

From scraping this website, we get an organized and informative dataframe that looks like this:


```python
dataf.head(10)
```

## Exploratory Data Analysis


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
dataf = pd.read_csv('original_data.csv')
dataf.pokemon = dataf.pokemon.str.lower()
dataf_stats = dataf.drop(['types'], axis=1)

dataf_long_stats = pd.melt(dataf_stats, id_vars = ['pokemon','type'], var_name = 'stats')
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
      <th>stats</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bulbasaur</td>
      <td>Grass</td>
      <td>hp</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ivysaur</td>
      <td>Grass</td>
      <td>hp</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>venusaur</td>
      <td>Grass</td>
      <td>hp</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>charmander</td>
      <td>Fire</td>
      <td>hp</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>charmeleon</td>
      <td>Fire</td>
      <td>hp</td>
      <td>58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charizard</td>
      <td>Dragon</td>
      <td>hp</td>
      <td>78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>squirtle</td>
      <td>Water</td>
      <td>hp</td>
      <td>44</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wartortle</td>
      <td>Water</td>
      <td>hp</td>
      <td>59</td>
    </tr>
    <tr>
      <th>8</th>
      <td>blastoise</td>
      <td>Water</td>
      <td>hp</td>
      <td>79</td>
    </tr>
    <tr>
      <th>9</th>
      <td>caterpie</td>
      <td>Bug</td>
      <td>hp</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



    The shape of the original dataframe is:  (807, 10)
    The shape of the melted dataframe is:  (5649, 4)


We can see that the new dataframe is 807*6=4842. Instead of having a "wider" dataframe, the dataframe is now "longer," which makes it easier to perform aggregate computations and visualizations.


```python
plt.figure(figsize=(15,13))
sns.swarmplot(x='stats', y = 'value', hue = 'type', data = dataf_long_stats)
plt.legend(loc = 2, bbox_to_anchor=(1,1))
```


```python
g = sns.factorplot(x='stats', y = 'value', data = dataf_long_stats,
                   col = 'type', col_wrap=5, kind = 'bar', size = 2)
```

Summary Statistics for each Stat for each type


```python
dataf_long_stats_gb = dataf_long_stats.groupby(['type', 'stats'])
dataf_long_stats_gb.aggregate([np.mean, np.std])
```

Now time for the fun part: trying to predict the types with these features!


```python
from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier()
clf = GridSearchCV(estimator = knn_clf, param_grid=dict(n_neighbors = list(range(1,25))))
clf.fit(trainX, trainy)
print(clf.best_estimator_) # best n_neighbors = 10

knn_clf = neighbors.KNeighborsClassifier(n_neighbors = 23)
knn_clf.fit(trainX, trainy)
print('knn classification rate:', str(sum(knn_clf.predict(testX) == testy)/len(testy)))

```

## Obtaining the Main Color of each Pokemon


```python
import requests
import PIL
import io
import numpy as np
import cv2
```


```python
two_colors = []
big_list = []
res3 = res2.tolist()

for i in res3:
    big_list.extend(i)
for i in big_list:
    in_list = i in two_colors
    if in_list == False: two_colors.append(i)
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

    100%|██████████| 807/807 [01:04<00:00, 12.53it/s]



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

The pokemon are in the same space as the previous graph. The difference here is that the color of the points changed to represent the pokemon's type. Since there are so many different types in such a small plot, it's difficult to see clear clusters. However, there are distinct clusters; for example, you can see that fire types and water types are on opposite ends. Just out of curiousity I'll try different machine learning methods to see how useful of a feature color is.

## Building a Predictive Model

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


Let's try out a few different machine learning algorithms to see which ones perform well by looking at their cross validation scores


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
results_df.sort_values(by = 'Mean Classification Rate', ascending=False)
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
      <td>0.304617</td>
      <td>0.025308</td>
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

    The cross-validation scores for this neural network are:  [0.26976744 0.32558139 0.29767442]
    The average cross-validation score for this neural network is:  0.2976744173802146



```python
nnet.fit(X_train, y_train_cat, epochs = 100, verbose=0, batch_size=32)

pred = nnet.predict(X_test)
#pred
accuracy_score(np.asarray(y_test_cat).argmax(axis=1),pred)
```




    <keras.callbacks.History at 0x7f9028c6f9b0>






    0.35185185185185186



## Conclusion
