---
layout: projects

---

**Introduction to K-Means Clustering**
======================================

<p style="text-align:center;">
<em>Kenny Lov</em><br><br>
</p>
First, let's used a contrived *toy* example to better understand this topic. k-means clustering works better if the clusters are spherical and normally distributed. For this example, we'll create a small, arbitrary dataset with 5 different clusters (5 populations with different means and variances).

``` r
# first create individual clusters with different distribution parameters
set.seed(123) # setting a seed for reproducibility

x1 <- rnorm(50, mean = 5, sd = 1)
y1 <- rnorm(50, mean = 10, sd = 1)
df1 <- data.frame(x = x1, y = y1, label = 1)

x2 <- rnorm(50, mean = 1.5, sd = 1.5)
y2 <- rnorm(50, mean = 3, sd = 1)
df2 <- data.frame(x = x2, y = y2, label = 2)

x3 <- rnorm(50, mean = 10, sd = 0.5)
y3 <- rnorm(50, mean = 5, sd = 2)
df3 <- data.frame(x = x3, y = y3, label = 3)

combined <- rbind(df1, df2, df3)
combined$label <- as.factor(combined$label) # label must be converted into a factor since it will be interpreted as a continuous variable, which it is not.
```

Now that we've created this *toy* dataset, let's visualize it and confirm that we've indeed created distinct clusters.

``` r
library(ggplot2)
th <- theme_linedraw() # setting the theme for the plots


tiff('./images/plot1.tiff', units="in", width=5, height=5, res=300)

ggplot(combined, aes(x= x, y = y)) +
  geom_point(aes(color=label), size = 2) +
  labs(title = "Toy Example") + 
  th

dev.off()
```

    ## png 
    ##   2

![]('./images/plot1.tiff')

Yes, there are indeed distinct clusters with various normal distributions! R already comes with a great built-in function `kmeans` that can compute clusters. However, for the sake of understanding, we'll hand-code a function that can compute the clusters as well as keep track of the data for each iteration to visualize the progress of the algorithm. If this sounds confusing now, it will make sense in a bit.

``` r
my_kmeans <- function(df, n_clusters){ # the function will take a dataframe and num clusters as input
  # first, get the range of possible values to initiate random centers
  
  Z_hist <- data.frame()          # create history of Z and centers to see                                                                    progress of iterations
  C_hist <- data.frame()
  
  Z <- rep(-1, nrow(df))     # Z are our indicator variable, we need to set placeholders for these                                           variables that are different values. values don't matter as long as they                                       are different from each other
  Z_new <- rep(0, nrow(df))  # these variables will tell the algorithm when to stop iterations
  
  centers <- array(0, dim = c(n_clusters, ncol(df)) ) # create a placeholder for centers array
  # now we can initialize random centers

  min <- min(df[,1])
  max <- max(df[,2])  
  for(row in seq(1, n_clusters)){
    for(col in seq(1, dim(centers)[2] ))
      centers[row, col] <- runif(1, min, max)
  }
  
  # now that we have the centers, we need to find differences between each point from each cluster
  # we will create a distance matrix
  
  Dist <- array(0, dim = c(nrow(df), n_clusters ) )
  
  iteration = 0 # keep track of iterations
  while(sum(Z-Z_new) !=0 ){ # keep iterating until Z and Z_new are equal
    
    Z_add <- data.frame(Z = Z_new, iteration = iteration)
    C_add <- data.frame(centers, iteration = iteration)
    
    Z_hist <- rbind(Z_hist, Z_add) # appending new iterations to keep track of the history
    C_hist <- rbind(C_hist, C_add)
    
    Z <- Z_new
    for(center in seq(1, nrow(centers))){
      distance <- apply(df, 1, function(x) sum((x - centers[center, ])^2) ) # compute euclidian distance 
      for(x in seq(1, length(distance))){
        Dist[x, center] <- distance[x] # filling in distance matrix with euclidian distances
      }
    }
    
    Z_new <- max.col(-Dist) # note that max.col function finds the column that has the maximum value.
                            # since we want to find the minimum distance, we invert by distance matrix
                            # by multiplying the whole matrix by -1. 
    
    # next, we need to move the centers since the center values for each cluster has changed
    
    for(center in seq(1, n_clusters)){
      for(var in seq(1, ncol(centers) )){
        if(sum(Z_new==center) ==0 ) centers[center, var] = centers[center, var]
        else centers[center, var] <- mean(df[Z_new == center, var])
      }
    }

    iteration = iteration + 1
  }
  # cat('Took', iteration - 1, 'iterations to converge!')
  Z_hist$prediction <- as.factor(Z_hist$Z)
  output <- list(Z_new, centers, Z_hist, C_hist)
  return(output)
}
```

Might not be the most efficient code possible with all the for loops and what not... but let's see what it can do.

Since this is an iterative approach, we can visualize the progress at every iteration using the history variables provided by my function!

``` r
library(gganimate)

no_labs = combined[,1:2]

prediction <- my_kmeans(no_labs, 3)
Z_hist <- cbind(combined, prediction[[3]])
c_hist <- prediction[[4]]

g <- ggplot(Z_hist, aes(x = x, y = y)) +
  geom_point(aes(color = prediction, shape = label), size = 2) +
  geom_point(data = c_hist, aes(x = X1, y = X2), size = 3) +
  labs(title = 'Iteration: {frame_time}') +
  th+
  transition_time(iteration)


animation::ani.options(ani.width= 1000, ani.height=1000, ani.res = 2000)

animate(g, nframes=  length(unique(Z_hist$iteration)), fps = 1)
```

![](images/unnamed-chunk-6-1.gif)

This animation essentially shows each step the algorithm takes to make its decision of which points are closest to each centroid. As you can see, the centroids (black dots) move around the grid and each color represents which centroid/cluster the individual samples are currently part of. The shape of each point represents the real group the point came from. We can see that the algorithm does a really good job in finding the centers for each group that we manually created, although there are some that are wrongly grouped.

Final Result (as a sanity check, let's compare with R's built in `kmeans` function):

``` r
combined$predicted <- as.factor(prediction[[1]]) # remeber to convert the integer values to  factors
centers <- prediction[[2]]

kmeans <- kmeans(no_labs, 3)

combined$kmeans_pred <- as.factor(kmeans$cluster)

g1 <- ggplot(combined, aes(x = x, y = y)) + 
  geom_point(aes(color = predicted, shape = label), size = 2) +
  geom_point(data = data.frame(centers), aes(X1, X2), size = 3) +
  labs(title = 'My K-means Prediction') +
  th

g2 <- ggplot(combined, aes(x=x , y = y)) +
  geom_point(aes(color = kmeans_pred, shape = label), size = 2) +
  geom_point(data = data.frame(data.frame(kmeans$centers)), aes(x, y), size = 3) +
  labs(title = "R's K-means Prediction") +
  th 

library(gridExtra) # import library to display graphs in a grid

tiff('./images/plot2.tiff', units="in", width=10, height=5, res=300)

grid.arrange(g1, g2, nrow=1, respect=TRUE)

dev.off()
```

    ## png 
    ##   2

![]('./images/plot2.tiff)

Even though some clusters are different colors, the points are actually clustered the same in both `my_kmeans` and R's `kmeans`.

Now let's compare the centers that the algorithm found to the actual centers that we created.

``` r
predicted_centers <- tail(c_hist, 3)[,-3] # k-means predicted centers
predicted_centers$real <- 'predicted'

real_centers <- matrix(c(5,10,1.5,3,10,5), byrow = TRUE, ncol=2) # generated data from these centers
real_centers <- data.frame(real_centers)
real_centers$real <- 'real'

both_centers <- rbind(predicted_centers, real_centers)

tiff('./images/plot3.tiff', units="in", width=5, height=5, res=300)
 
ggplot(combined, aes(x = x, y = y)) +
  geom_point(aes(color = label)) +
  geom_point(data = both_centers, aes(x= X1,y= X2, shape = real), size = 3) +
  labs(title = 'Comparing Real and Predicted Centers') +
  th

dev.off()
```

    ## png 
    ##   2

![]('./images/plot3.tiff')

Although the predicted cluster centers are not perfectly on top of the real centers (due to the random nature of the sampling), they are very close to each other, showing that the algorithm does work when there are distinct clusters!

Since in this case our labels are known, we can caclulate the confusion matrix for the prediction of this algorithm.

Now, since this is a *boring* example, let's use a more interesting dataset!
