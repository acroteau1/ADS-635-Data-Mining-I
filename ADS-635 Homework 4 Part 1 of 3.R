#######################################################
# Homework 4 Question 1
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/24/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(rpart) # trees
library(gbm) # boosting
library(randomForest) # RF / bagging
library(ISLR)
library(class) #kNN

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import at data
wage_data <- read.csv("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\wage.csv", header = T)

#######################################################
# Variable Information
# 3000 observations of 11 variables
# year - int - Year that wage information was recorded
#
# age - int - age of worker
#
# maritl - chr - A factor with levels 1. Never Married 
# 2. Married 3. Widowed 4. Divorced and 
# 5. Separated indicating marital status
#
# race - chr - A factor with levels 1. White 2. Black 
# 3. Asian and 4. Other indicating race
#
# education - chr - A factor with levels 1. < HS Grad 
# 2. HS Grad 3. Some College 4. College Grad and 
# 5. Advanced Degree indicating education level
#
# region - chr - Region of the country (mid-atlantic only)
#
# jobclass - chr - A factor with levels 1. Industrial and 
# 2. Information indicating type of job
#
# health - chr - A factor with levels 1. <=Good and 
# 2. >=Very Good indicating health level of worker
#
# health_ins - chr - A factor with levels 1. Yes and 
# 2. No indicating whether worker has health insurance
#
# logwage - num - log of worker's wage
#
# wage - num - worker's raw wage
#######################################################

## 1a - Apply bagging, boosting, and random forests to a 
## data set of your choice (not one used in the committee 
## machines labs). Fit the models on a training set and 
## evaluate them on a test set.

# Make a binary variable
High <- ifelse(wage_data$wage <= 60, "No", "Yes")
my_wage <- data.frame(wage_data[,-11], High)
my_wage$High <- ifelse(my_wage$High == "Yes", 1, 0)

# Create a training set and test set for the data
set.seed(123) # Setting seed for reproducibility
sample_index <- sample(1:nrow(my_wage), nrow(my_wage)*0.7)
train_data <- my_wage[sample_index, ]
test_data <- my_wage[-sample_index, ]

# Grow a single tree
model.control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
fit <- rpart(High ~ age + maritl + race + education + 
               jobclass + health + health_ins, data = train_data, 
             method = "class", control = model.control)

# Model the full single tree
x11()
plot(fit)
text(fit, use.n = TRUE, cex = .5)

# Prune the tree back
min_cp = which.min(fit$cptable[,4])
x11()
plot(fit$cptable[,4], main = "Cp for model selection", ylab = "cv error")

pruned_fit <- prune(fit, cp = fit$cptable[min_cp, 1])
x11()
plot(pruned_fit)
text(pruned_fit, use.n = TRUE, cex = .5)

# Compute test error for a single tree
my_pred <- predict(pruned_fit, newdata = test_data, type = "class")
y_true <- as.numeric(test_data$High)-1 
y_hat <- as.numeric(my_pred)-1
misclass_tree <- sum(abs(y_true- y_hat))/length(y_hat)
misclass_tree # 1.045556

# Random Forest 
rf.fit <- randomForest(High ~ age + maritl + race + education + 
                         jobclass + health + health_ins, 
                       data = train_data, n.tree = 10000)
x11()
varImpPlot(rf.fit)
importance(rf.fit)
summary(rf.fit)

y_hat <- predict(rf.fit, newdata = test_data, type = "response")
y_hat <- as.numeric(y_hat)-1
misclass_rf <- sum(abs(y_true - y_hat))/length(y_hat)
misclass_rf # 0.07899703

# Bagging
bag.fit <- randomForest(High ~ age + maritl + race + education +
                          jobclass + health + health_ins, 
                        data = train_data, n.tree = 10000, mtry = 2)
x11()
varImpPlot(bag.fit)
importance(bag.fit)

y_hat <- predict(bag.fit, newdata = test_data, type = "response")
y_hat <- as.numeric(y_hat)-1
misclass_bag <- sum(abs(y_true - y_hat))/length(y_hat)
misclass_bag # 0.07887012

# Boosting
boost.train <- train_data
boost.test <- test_data

boost.train$maritl <- as.factor(boost.train$maritl) # Convert to factor
boost.test$maritl <- as.factor(boost.test$maritl) # Convert to factor
boost.train$race <- as.factor(boost.train$race) # Convert to factor
boost.test$race <- as.factor(boost.test$race) # Convert to factor
boost.train$education <- as.factor(boost.train$education) # Convert to factor
boost.test$education <- as.factor(boost.test$education) # Convert to factor
boost.train$jobclass <- as.factor(boost.train$jobclass) # Convert to factor
boost.test$jobclass <- as.factor(boost.test$jobclass) # Convert to factor
boost.train$health <- as.factor(boost.train$health) # Convert to factor
boost.test$health <- as.factor(boost.test$health) # Convert to factor
boost.train$health_ins <- as.factor(boost.train$health_ins) # Convert to factor
boost.test$health_ins <- as.factor(boost.test$health_ins) # Convert to factor

boost.fit <- gbm(High ~ age + maritl + race + education +
                 jobclass + health + health_ins, 
                 data = boost.train, n.trees = 1000, shrinkage = .1, 
                 interaction.depth = 3, distribution = "adaboost")
boost.fit2 <- gbm(High ~ age + maritl + race + education +
                  jobclass + health + health_ins, 
                  data = boost.train, n.trees = 1000, shrinkage = .6, 
                  interaction.depth = 3, distribution = "adaboost")
summary(boost.fit)

y_hat <- predict(boost.fit, newdata = test_data, type = "response")
y_hat <- as.numeric(y_hat)-1
misclass_boost <- sum(abs(y_true - y_hat))/length(y_hat)
misclass_boost # 0.04687535

## 1b - How accurate are these results compared to more simplistic 
## (non-ensemble) methods (e.g., logistic regression, kNN, etc)? 

# Logistic Regression
lg.fit <- glm(High ~ age + maritl + race + education +
                jobclass + health + health_ins, 
              data = train_data, family = binomial)
summary(lg.fit)

# Compute the test and training error
pred_train <- predict(lg.fit, newdata = train_data, type = "response")
y_hat_train <- round(pred_train)
train_err <- length(which(train_data$High != y_hat_train))/length(y_hat_train)

pred_test <- predict(lg.fit, newdata = test_data, type = "response")
y_hat_test <- round(pred_test)
test_err <- length(which(test_data$High != y_hat_test))/length(y_hat_test)

# LR results
train_err # 0.04809524
test_err # 0.04555556

# kNN
train_data_knn <- my_wage[sample_index, -ncol(my_wage)] # Excluding High column
test_data_knn <- my_wage[-sample_index, -ncol(my_wage)] # Excluding High column
# Extracting target values
train_labels <- my_wage$High[sample_index]
test_labels <- my_wage$High[-sample_index]

# Convert character columns to factors
char_columns <- c("maritl", "race", "education", "jobclass", "health", "health_ins")
train_data_knn[char_columns] <- lapply(train_data_knn[char_columns], as.factor)
test_data_knn[char_columns] <- lapply(test_data_knn[char_columns], as.factor)

# Convert factors to numeric
train_data_knn[char_columns] <- lapply(train_data_knn[char_columns], as.numeric)
test_data_knn[char_columns] <- lapply(test_data_knn[char_columns], as.numeric)

# Drop the 'region' column as it may not be informative
train_data_knn$region <- NULL
test_data_knn$region <- NULL

pr <- knn(train_data_knn, test_data_knn, cl = train_labels, k= 1)

# Create confusion matrix
tab <- table(pr,test_labels)
tab
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
mcr <- (1 - (accuracy(tab)/100))
mcr