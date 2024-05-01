#######################################################
# Homework 1
# ADS-635: Data Mining I
# Alison Croteau
# Created: 9/14/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
# install.packages("dplyr")
# install.packages("Hmisc")
# install.packages("leaps")
# install.packages("stats")
# install.packages("caret")
# install.packages("class")
library(dplyr)
library(Hmisc)
library(leaps)
library(stats)
library(caret)
library(class)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import data
smarket <- read.csv("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\Smarket.csv", header = T)
hitters <- read.csv("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\Hitters.csv", header = T)

# 1a - What are the dimensions of this data?
dim(smarket)

# 1b - What are the averages for: Lag1 – Lag5? Calculate this in two different ways.
# 1b Method 1 - use the mean() function
mean(smarket$Lag1)
mean(smarket$Lag2)
mean(smarket$Lag3)
mean(smarket$Lag4)
mean(smarket$Lag5)

# 1b Method 2 - Using the summary() function
summary(smarket$Lag1)
summary(smarket$Lag2)
summary(smarket$Lag3)
summary(smarket$Lag4)
summary(smarket$Lag5)

# 1c - What type of variable is Direction? Using the table function, 
# comment on the frequency of Up and Down.
print(typeof(smarket$Direction))
table(smarket$Direction)

# 1d - Create a list object with data.frames for 2001, 2002, 2003, 2004 and 2005.
# 1d - Split the data into data frames by year and store them in a list
year_list <- split(smarket, smarket$Year)

# 1d - Access the data by year
year_2001 <- year_list[[1]]
year_2002 <- year_list[[2]]
year_2003 <- year_list[[3]]
year_2004 <- year_list[[4]]
year_2005 <- year_list[[5]]

# 2 - Pre-process/clean the data
dim(hitters)
head(hitters)

# Remove rows with missing observations
hitters <- na.omit(hitters)

#Plot data to check for outliers
plot(hitters$AtBat)
plot(hitters$Hits)
plot(hitters$HmRun)
plot(hitters$Runs)
plot(hitters$RBI)
plot(hitters$Walks)
plot(hitters$Years)
plot(hitters$PutOuts)
plot(hitters$Assists)
plot(hitters$Errors)
plot(hitters$Salary)

# Output data as RData
save(hitters, file = "Hitters.RData")

# 2 - Investigate the data using exploratory data analysis
plot(hitters$Salary, hitters$AtBat, col = "blue", varwidth = T, xlab = "Times at Bat", ylab = "Salary")
plot(hitters$Salary, hitters$Hits, col = "blue", varwidth = T, xlab = "Number of Hits", ylab = "Salary")
plot(hitters$Salary, hitters$HmRun, col = "blue", varwidth = T, xlab = "Home Runs", ylab = "Salary")
plot(hitters$Salary, hitters$Runs, col = "blue", varwidth = T, xlab = "Runs", ylab = "Salary")
plot(hitters$Salary, hitters$RBI, col = "blue", varwidth = T, xlab = "Runs Batted", ylab = "Salary")
plot(hitters$Salary, hitters$Walks, col = "blue", varwidth = T, xlab = "Walks", ylab = "Salary")
plot(hitters$Salary, hitters$Years, col = "blue", varwidth = T, xlab = "Years", ylab = "Salary")
plot(hitters$Salary, hitters$PutOuts, col = "blue", varwidth = T, xlab = "Put Outs", ylab = "Salary")
plot(hitters$Salary, hitters$Assists, col = "blue", varwidth = T, xlab = "Assists", ylab = "Salary")
plot(hitters$Salary, hitters$Errors, col = "blue", varwidth = T, xlab = "Errors", ylab = "Salary")

# 3 - Divide your data from Q2 into training and test.
# Take salaries under $150k data as the test set, and pool the rest for training
grab <- which(hitters$Salary <= 150)
grab
my_test <- hitters[grab, ]
my_train <- hitters[-grab, ]
dim(my_test)
dim(my_train)
X_train <- my_train[,-5]
Y_train <- my_train[,5]
X_test <- my_test[,-5]
Y_test <- my_test[,5]

# 3 - Use k-nearest neighbors to predict salary. 
# Define the number of neighbors (k) you want to test and 
# create a vector to store MSE per each k
set.seed(1)
k_values <- 1:15
mse_values <- numeric(length(k_values))

# Perform k-nearest neighbors regression for each k
for (k in k_values) {
  # Train the model
  model <- train(Salary ~ ., data = my_train, method = "knn", tuneGrid = data.frame(k = k))
  # Make predictions on the test set
  predictions <- predict(model, newdata = my_test)
  # Calculate the mse and store
  mse <- mean((my_test$Salary - predictions)^2)
  mse_values[k] <- mse
}

# Plot the MSE values for different k values
plot(k_values, mse_values, xlab = "Number of Neighbors (k)", ylab = "MSE", 
     type = "b")

# Determined k = 3 is ideal. Now utilize that in knn()
attach(hitters)

# divide the data into test and training
N = length(hitters[,1])

set.seed(1)
my_train <- sample(1:N, size = (2/3)*N, replace = FALSE)

train <- hitters[my_train, ]
test <- hitters[-my_train, ]

X_train <- train[ ,-5]
Y_train <- train[ ,5]

X_test <- test[,-5]
Y_test <- test[,5]

# Training error for k = 3
Y_train_hat <- knn(X_train, X_train, Y_train, k= 3)
data.frame(Y_train, Y_train_hat)
table(Y_train, Y_train_hat)

which(Y_train != Y_train_hat) 
length(which(Y_train != Y_train_hat)) 
(1/length(Y_train))*length(which(Y_train != Y_train_hat))

# test error for k = 3
Y_test_hat <- knn(X_train, X_test, Y_train, k= 3)
data.frame(Y_test, Y_test_hat)
table(Y_test, Y_test_hat)

# 4a - Load in the Boston data set. Read about the data set:
# How many rows are in this data set? How many columns? 
# What do the rows and columns represent?
Boston <- read.csv("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\Boston.csv",
                   header = T)
?Boston
dim(Boston)
  
# 4b - Make some pairwise scatterplots of the predictors (columns) in this 
# data set. Describe your findings.
# Compare median home value based on some of the other variables
attach(Boston)
plot(medv, crim, 
     xlab = "Median Home Value in $1000s", 
     ylab = "Town-Wise Per Capita Crime Cate")
plot(medv, tax, 
     xlab = "Median Home Value in $1000s",
     ylab = "Property Tax Rate")
plot(medv, rad, 
     xlab = "Median Home Value in $1000s",
     ylab = "Radial Highway Accessability")
plot(medv, dis, 
     xlab = "Median Home Value in $1000s",
     ylab = "Employment Centre Distance")

# 4c - Are any of the predictors associated with per capita crime rate? 
# If so, explain the relationship.
# Perform a linear regression
crimrate <- lm(crim ~ ., data = Boston)
crimsum <- summary(crimrate)
crimsum

# 4d - Do any of the census tracts of Boston appear to have particularly 
# high crime rates? Tax rates? Pupil-teacher ratios? Comment on the range 
# of each predictor.
# Start by using summary() to see distributions.
summary(crim)
summary(tax)
summary(ptratio)

# Create a data frame with values above 3rd-quartile values
high_crime <- Boston[Boston$crim > 3.67708,]
high_tax <- Boston[Boston$tax > 666.0,]
high_ratio <- Boston[Boston$ptratio > 20.20,]

# Get dimensions and view a snip of each high-value 
dim(high_crime)
head(high_crime)
dim(high_tax)
head(high_tax)
dim(high_ratio)
head(high_ratio)

# Get ranges
crim_range <- range(Boston$crim)
tax_range <- range(Boston$tax)
ratio_range <- range(Boston$ptratio)
crim_range
tax_range
ratio_range

# 4e - How many of the census tracts in this data set bound the Charles river?
chas_bound <- sum(Boston$chas == 1)
chas_bound
  
# 4f - What is the median pupil-teacher ratio among the towns in this data set?
summary(Boston$ptratio)

# 4g - Which census tract of Boston has the lowest median value of owner occupied 
# homes? What are the values of the other predictors for that census tract, and how 
# do those values compare to the overall ranges for those predictors? Comment on 
# your findings.
min_medv <- which.min(Boston$medv)
min_medv_obs <- Boston[min_medv,]
min_medv_obs
summary(Boston)

# 4h - In this data set, how many of the census tracts average more than seven 
# rooms per dwelling? More than eight rooms per dwelling? Comment on the census 
# tracts that average more than eight rooms per dwelling.
rooms_7 <- Boston[Boston$rm > 7,]
dim(rooms_7)
rooms_8 <- Boston[Boston$rm > 8,]
dim(rooms_8)
summary(rooms_8)