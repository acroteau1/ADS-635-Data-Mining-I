#######################################################
# This code utilizes kNN for different values of k
#
# Rachael Hageman Blair
# Created: 8/1/2012
# Modified: 6/2020
#######################################################

rm(list = ls())
setwd("~/Dropbox/STA_545_Fall2019/Comp_Labs")

#install.packages("ggplot2")
library("ggplot2")

##################################
#  Load and visualize the data
#  
##################################
data <- read.delim("clipped_data.txt", sep = "\t", header= FALSE)
train <- data.frame(X1 = data[,1], X2 = data[,2], Y = data[,3])
dim(train)
head(train)
g <- ggplot(train, aes(X1,X2)) + geom_point(aes(colour = as.factor(Y))) + theme(legend.position = "none")
quartz()
plot(g)
ggsave(filename = "orig.png", plot = g, height = 5, width = 5)

######################################################
#  Create a test data set .... 
#  a grid of values spanning the ranges of X1 and X2
#  (long term goal: visualization)
######################################################
minX1 <- min(train$X1)
minX2 <- min(train$X2)
maxX1 <- max(train$X1)
maxX2 <- max(train$X2)

# ranges
X1.range <- seq(from = minX1, to = maxX1, length.out = 100)
X2.range <- seq(from = minX2, to = maxX2, length.out = 100)

# Create the test set
test <- data.frame(X1 = rep(X1.range, 100), X2 = rep(X2.range, each = 100))
g2 <- ggplot(test, aes(X1,X2)) + geom_point(size = 0.5)
quartz()
plot(g2)

################################
## Try different values of k
################################
require(class)
knnplot <- function(train, test, k){
	 KNN <- knn(train[, c('X1', 'X2')], test, train$Y, k)
	 test$predict <- KNN
	 
	 # change factor to numeric
	 test$z <- c(0, 1)[sapply(test$predict, as.numeric)]
	
	 title = paste("k=", as.character(k), sep ="")
	 
	 g <- ggplot(data = test, aes(X1,X2)) + geom_point(aes(colour = predict), size = 0.5) + geom_contour(aes(z=z), colour = 'black', size = 0.1) + theme(legend.position = "none") + labs(title = title)
	
	#add the training points in
	g <- g + geom_point(data = train, aes(X1,X2,colour = as.factor(Y), shape = 'x'))

	return(g)

}

###############################################
## Try differnt values of k, and save
###############################################
filer <- paste("k", c(1:10), ".png", sep="")
for (i in 1:10){
	p <- knnplot(train, test, i)
	ggsave(filename = filer[i], plot = p, height = 5, width = 5)
}





### Psuedo code

# done
my_train
my_test

# create a loop to look at kNN for different values of k
k_vals <- c(1, 3, 5, 7, 9, 11)
store_error <- c()
for (i in 1:6){
    
    # select "k"
    kk <- k_vals[i]
    
    # apply the algorithm
    fit <- knn(my_train, my_test, my_train[,1], k = kk)
    
    # check the answer
    fit
    
    # make a call --- you may have 2.1 2.7 ----> "round"
    # ?round
    
    # calculate the error
    which(fit != my_test[,1])
    
}




















































