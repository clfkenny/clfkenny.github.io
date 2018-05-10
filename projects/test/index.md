2.4.8
=====

Question 1: Does being a public/private school affect the graduation
rate?

![](Homework_3_files/figure-markdown_strict/unnamed-chunk-2-1.png)


        Welch Two Sample t-test

    data:  College$Grad.Rate by College$Private
    t = -10.579, df = 431.97, p-value < 2.2e-16
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
     -15.36276 -10.54880
    sample estimates:
     mean in group No mean in group Yes 
             56.04245          68.99823 

-   Based on the t-test, it does appear that there is a significant
    difference in graduation between private vs public schools. Private
    schools on average have about a 13% higher chance in graduation
    rate.

Question 2: Does increasing the student to faculty ratio decrease the
percentage of students who donate? Maybe a lower S:F ratio might give
students a better learning experience, and thus more willing to donate.
![](Homework_3_files/figure-markdown_strict/unnamed-chunk-3-1.png)


    Call:
    lm(formula = perc.alumni ~ S.F.Ratio, data = College)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -27.939  -8.407  -1.146   7.116  34.836 

    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  40.5165     1.5062   26.90   <2e-16 ***
    S.F.Ratio    -1.2614     0.1029  -12.26   <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 11.35 on 775 degrees of freedom
    Multiple R-squared:  0.1624,    Adjusted R-squared:  0.1613 
    F-statistic: 150.2 on 1 and 775 DF,  p-value: < 2.2e-16

-   Although the model doesn’t fit the data too well based on the R^2
    value, the S:F ratio appears to be a significant predictor in the
    percentage of alumni who donate. There is indeed a negative trend,
    meaning that a lower S:F ratio on average results in more alumni who
    donates.

\#4.7.1

\#4.7.4 \#\# a) 10% of observations will be used, since uniformally
distributed.

b)
--

1% of total observations will be used. 10% of observations of
*X*<sub>1</sub> will be within the range. Within 10%, only 10% will be
within the range for *X*<sub>2</sub>. Thus 10% \* 10% = 1% of the total
observations.

c)
--

Only 0.1<sup>100</sup> of the observations will be used.

d)
--

Based on my answers, it seems that for a uniform distribution, for every
predictor added, the total number of observations used will be
multiplied by 0.1.

e)
--

The length of the hypercube will correspond to the number of
observations used to make the prediction. Thus the number of
observations used will be 0.1<sup>*p*</sup> where p is the number of
predictors.

-   For p = 1: 0.1<sup>1</sup>

-   for p = 2: 0.1<sup>2</sup>

-   for p = 100: 0.1<sup>100</sup>

\#4.7.6 \#\# a)

    b0 <- -6
    b1 <- 0.05  # hours studied
    b2 <- 1  # undergrad GPA

    prob <- exp(b0 + b1 * 40 + b2 * (3.5))/(1 + exp(b0 + b1 * 40 + 
        b2 * (3.5)))
    cat("The probability that a student who studies for 40h and has an undergrad GPA of 3.5 has a ", 
        prob * 100, "% \nchance of getting an A", sep = "")

    The probability that a student who studies for 40h and has an undergrad GPA of 3.5 has a 37.75407% 
    chance of getting an A

b)
--

    logit <- log(0.5/(1 - 0.5))
    hours <- (logit - b0 - b2 * (3.5))/b1
    cat("The student will need to study ", hours, " hours to have a 50% chance of getting an A in the class", 
        sep = "")

    The student will need to study 50 hours to have a 50% chance of getting an A in the class

\#4.7.10 \#\# a)

          Year           Lag1               Lag2               Lag3         
     Min.   :1990   Min.   :-18.1950   Min.   :-18.1950   Min.   :-18.1950  
     1st Qu.:1995   1st Qu.: -1.1540   1st Qu.: -1.1540   1st Qu.: -1.1580  
     Median :2000   Median :  0.2410   Median :  0.2410   Median :  0.2410  
     Mean   :2000   Mean   :  0.1506   Mean   :  0.1511   Mean   :  0.1472  
     3rd Qu.:2005   3rd Qu.:  1.4050   3rd Qu.:  1.4090   3rd Qu.:  1.4090  
     Max.   :2010   Max.   : 12.0260   Max.   : 12.0260   Max.   : 12.0260  
          Lag4               Lag5              Volume       
     Min.   :-18.1950   Min.   :-18.1950   Min.   :0.08747  
     1st Qu.: -1.1580   1st Qu.: -1.1660   1st Qu.:0.33202  
     Median :  0.2380   Median :  0.2340   Median :1.00268  
     Mean   :  0.1458   Mean   :  0.1399   Mean   :1.57462  
     3rd Qu.:  1.4090   3rd Qu.:  1.4050   3rd Qu.:2.05373  
     Max.   : 12.0260   Max.   : 12.0260   Max.   :9.32821  
         Today          Direction 
     Min.   :-18.1950   Down:484  
     1st Qu.: -1.1540   Up  :605  
     Median :  0.2410             
     Mean   :  0.1499             
     3rd Qu.:  1.4050             
     Max.   : 12.0260             

                  Year         Lag1        Lag2        Lag3         Lag4
    Year    1.00000000 -0.032289274 -0.03339001 -0.03000649 -0.031127923
    Lag1   -0.03228927  1.000000000 -0.07485305  0.05863568 -0.071273876
    Lag2   -0.03339001 -0.074853051  1.00000000 -0.07572091  0.058381535
    Lag3   -0.03000649  0.058635682 -0.07572091  1.00000000 -0.075395865
    Lag4   -0.03112792 -0.071273876  0.05838153 -0.07539587  1.000000000
    Lag5   -0.03051910 -0.008183096 -0.07249948  0.06065717 -0.075675027
    Volume  0.84194162 -0.064951313 -0.08551314 -0.06928771 -0.061074617
    Today  -0.03245989 -0.075031842  0.05916672 -0.07124364 -0.007825873
                   Lag5      Volume        Today
    Year   -0.030519101  0.84194162 -0.032459894
    Lag1   -0.008183096 -0.06495131 -0.075031842
    Lag2   -0.072499482 -0.08551314  0.059166717
    Lag3    0.060657175 -0.06928771 -0.071243639
    Lag4   -0.075675027 -0.06107462 -0.007825873
    Lag5    1.000000000 -0.05851741  0.011012698
    Volume -0.058517414  1.00000000 -0.033077783
    Today   0.011012698 -0.03307778  1.000000000

![](Homework_3_files/figure-markdown_strict/unnamed-chunk-6-1.png)![](Homework_3_files/figure-markdown_strict/unnamed-chunk-6-2.png)

b)
--


    Call:
    glm(formula = dummy ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, 
        data = Weekly)

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.7793  -0.5456   0.3891   0.4454   0.6527  

    Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  0.565983   0.021078  26.852   <2e-16 ***
    Lag1        -0.010042   0.006442  -1.559   0.1194    
    Lag2         0.014086   0.006469   2.177   0.0297 *  
    Lag3        -0.003752   0.006442  -0.582   0.5604    
    Lag4        -0.006655   0.006447  -1.032   0.3021    
    Lag5        -0.003404   0.006429  -0.529   0.5966    
    Volume      -0.005744   0.009041  -0.635   0.5254    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for gaussian family taken to be 0.2462898)

        Null deviance: 268.89  on 1088  degrees of freedom
    Residual deviance: 266.49  on 1082  degrees of freedom
    AIC: 1573.5

    Number of Fisher Scoring iterations: 2

The only predictor that appears statistically significant is Lag2

c)
--

<table>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">Down</th>
<th style="text-align: right;">Up</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Down</td>
<td style="text-align: right;">52</td>
<td style="text-align: right;">44</td>
</tr>
<tr class="even">
<td>Up</td>
<td style="text-align: right;">432</td>
<td style="text-align: right;">561</td>
</tr>
</tbody>
</table>

The confusion matrix tells us that there is a (52+561)/(52+561+44+432) =
0.5629017 proportion of correct predictions. The confusion matrix tells
us more information about the types of mistakes such as false positives
or false negatives. For example, our model predicted ‘Up’ 432 times when
the true classification is ‘Down’. Additionally, our model predicted
‘Down’ 44 times when the true classification is ‘Up’. This tells us that
our model is very likely to predict ‘Up’, even when the true
classification is ‘Down’ (i.e. a false positive if ‘Up’ is considered
positive).

d)
--

<table>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">Down</th>
<th style="text-align: right;">Up</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Down</td>
<td style="text-align: right;">9</td>
<td style="text-align: right;">5</td>
</tr>
<tr class="even">
<td>Up</td>
<td style="text-align: right;">34</td>
<td style="text-align: right;">56</td>
</tr>
</tbody>
</table>

The confusion matrix tells us that there is a (9+56)/(9+56+5+34) = 0.625
proportion of correct predictions.

e)
--

<table>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">0</th>
<th style="text-align: right;">1</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0</td>
<td style="text-align: right;">9</td>
<td style="text-align: right;">5</td>
</tr>
<tr class="even">
<td>1</td>
<td style="text-align: right;">34</td>
<td style="text-align: right;">56</td>
</tr>
</tbody>
</table>

0 corresponds to ‘Down’ while 1 corresponds to ‘Up’. The confusion
matrix tells us that there is a (9+56)/(9+56+5+34) = 0.625 proportion of
correct predictions.

g)
--

<table>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">Down</th>
<th style="text-align: right;">Up</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Down</td>
<td style="text-align: right;">21</td>
<td style="text-align: right;">30</td>
</tr>
<tr class="even">
<td>Up</td>
<td style="text-align: right;">22</td>
<td style="text-align: right;">31</td>
</tr>
</tbody>
</table>

The confusion matrix tells us that there is a (21+32)/(21+32+29+22) =
0.5096154 proportion of correct predictions.

h)
--

Both logistic regression and LDA produce the same proportion of correct
predictions and are higher than the proportion of correct predictions
from KNN, so either logistic regression or LDA will be the best fit for
this data.
