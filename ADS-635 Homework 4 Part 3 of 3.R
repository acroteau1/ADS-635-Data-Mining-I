#######################################################
# Homework 4 Question 3
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/24/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(rpart) # trees
library(gbm) # boosting
library(randomForest) # RF / bagging

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import the downloaded data file "spam.data"
spam <- read.table("spam.data", header = FALSE)
head(spam)

#######################################################
# Variable Information
# 4601 observations of 58 variables
# 57 continuous, 1 nominal class label
# 1813 Spam = 39.4%
# The last column of 'spambase.data' denotes whether the e-mail was 
# considered spam (1) or not (0), i.e. unsolicited commercial e-mail.  
# Most of the attributes indicate whether a particular word or
# character was frequently occuring in the e-mail. 
#######################################################

# Split the data into training and test sets
set.seed(123)
train_indices <- sample(1:nrow(spam), nrow(spam)*0.7)
train_data <- spam[train_indices, ]
test_data <- spam[-train_indices, ]

# Extract predictors and response
train_x <- train_data[, -58]
train_y <- as.factor(train_data[, 58])
test_x <- test_data[, -58]
test_y <- as.factor(test_data[, 58])

# Explore a range of values for m based on training data
m_values <- seq(1, ncol(train_x), by = 5)
oob_errors <- numeric(length(m_values))
test_errors <- numeric(length(m_values))

for(i in 1:length(m_values)){
  # Create a random forest
  rf <- randomForest(x = train_x, y = train_y, mtry = m_values[i], ntree = 100)
  
  # Store OOB error
  oob_errors[i] <- rf$err.rate[nrow(rf$err.rate), "OOB"]
  
  # Predict on the test data and record the test error
  preds <- predict(rf, newdata = test_x)
  test_errors[i] <- mean(preds != test_y)
}

# Table of results
spam_results <- data.frame(
  m = m_values, OOB_Error = oob_errors, Test_Error = test_errors)
spam_results

# Plot the results
plot(m_values, oob_errors, type = "l", col = "blue", ylim = 
       c(min(c(oob_errors, test_errors)), max(c(oob_errors, test_errors))),
     xlab = "Number of Variables at Each Split m", 
     ylab = "Error Rate",
     main = "OOB and Test Error Rates at m")
lines(m_values, test_errors, col = "red")
legend("topright", legend = c("OOB Error", "Test Error"), fill = c("blue", "red"))