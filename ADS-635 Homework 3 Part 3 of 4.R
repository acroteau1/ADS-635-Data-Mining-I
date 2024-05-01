#######################################################
# Homework 3 Question 3
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/17/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(datasets)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import and look at data
data(iris)
summary(iris)

#######################################################
# Variable Information
# 150 observations of 5 variables
# Sepal Length
# Sepal Width
# Petal Length
# Petal Width
# Species - Factor with 3 levels
#######################################################

# 3c - Write an *.R function to implement k-fold cross validation. 
# Apply it to a data set of your choice. Use it to compare the 
# results of 10-fold, 5-fold and LOOCV. Submit your fully commented 
# function, as well as the application of the dataset that you selected.

# Input:
# data: dataset
# k: number of folds; if k equals the number of data points, then it's LOOCV
# Outputs mean and standard deviation of the validation errors

kf_cv <- function(data, k) {
  n <- nrow(data) # get the rows from the dataset
  folds <- list() # list to contain the folds
  fold_size <- (n / k) # Stores the folds
  all_obs <- (1:n) # All observations
  results <- list() # list to contain the results
  
  for(i in 1:k) {
    # Randomly sample the fold size based on remaining observation
    remain <- sample(all_obs, fold_size, replace = FALSE)
    
    # Store indices
    folds[[i]] <- remain
    
    # For the last fold, take all remaining data points
    if (i == k){
      folds[[i]] <- all_obs}
    
    #update remaining indices to reflect what was taken out
    all_obs <- setdiff(all_obs, remain)
    all_obs
  }

for (i in 1:k){
  # Create the test and training sets
  indis <- folds[[i]]
  train <- data[-indis, ]
  test <- data[indis, ]
  
  # Linear Regression Model
  PL_model <- lm(as.numeric(Petal.Length) ~ ., data = train)
  pred <- predict(PL_model, newdata = test)
  RMSE <- sqrt(mean((test$Petal.Length - pred)^2))
  results[[i]] <- RMSE
}
return(results)
}

# Run on the iris data set
# 5-fold cross validation
five_fold <- kf_cv(iris, 5)
five_fold
mean(unlist(five_fold))

# 10-fold cross validation
ten_fold <- kf_cv(iris, 10)
ten_fold
mean(unlist(ten_fold))

# LOOCV
LOOCV <- kf_cv(iris, 149) # 149 = 150 - 1
LOOCV
mean(unlist(LOOCV))