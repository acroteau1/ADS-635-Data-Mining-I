#######################################################
# Homework 3 Question 2
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/15/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
# install.packages("rpart")
library("rpart")

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import and look at data
# load("C:/Users/acrot/Downloads/cleveland-2-1 (1).RData")
attach(cleveland)
head(cleveland)

#######################################################
# Variable Information
# 296 observations of 15 variables
# age - a numeric vector
# gender - a factor with levels fem male
# cp - a factor with levels abnang angina asympt notang
# trestbps - a numeric vector
# chol - a numeric vector
# fbs - a factor with levels fal true
# restecg - a factor with levels abn hyp norm
# thalach - a numeric vector
# exang - a factor with levels fal true
# oldpeak - a numeric vector
# slope - a factor with levels down flat up
# ca - a numeric vector
# thal - a factor with levels fix norm rev
# diag1 - a factor with levels buff sick
# diag2 - a factor with levels H S1 S2 S3 S4
#######################################################

# Fit a CART model and a logistic regression model to the Cleveland 
# heart-disease data. Compare the results, and comment on the performance.

# Fit the CART model - Classification Tree
model_control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
cart_model <- rpart(diag1 ~ age + gender + cp + trestbps + chol + fbs + 
                      restecg + thalach + exang + oldpeak + slope + ca
                    + thal, data = cleveland, method = "class", 
                    control = model_control)
print(cart_model)

# Plot the CART model - Classification Tree
x11()
plot(cart_model, uniform = T, compress = T, main = "CART Model")
text(cart_model, cex = 1)

# Prune the CART model
model_control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
fit_cleveland <- rpart(diag1 ~ age + gender + cp + trestbps + chol + fbs + 
                         restecg + thalach + exang + oldpeak + slope + ca
                       + thal, data = cleveland, method = "class", 
                       control = model_control)
names(fit_cleveland)

# Plot x-value error for model selection
x11()
plot(fit_cleveland$cptable[,4], main = "Xval error for model selection", ylab = "xerr")

# Plot Cp for model selection
x11()
plot(fit_cleveland$cptable[,1], main = "Cp for model selection", ylab = "Cp")
min_cp = which.min(fit_cleveland$cptable[,0.5])

# Prune the tree
pruned_fit_cle <- prune(fit_cleveland, cp = 0.03)

# Plot the pruned tree
x11()
plot(pruned_fit_cle, branch = 0.03, compress = T, main = "Pruned Tree")
text(pruned_fit_cle, cex = 0.5)

# Plot the full tree
x11()
plot(fit_cleveland, branch = 0.03, compress = T, main = "Full Tree")
text(fit_cleveland, cex = 0.5)

# Fit a Regression Tree
model_controls <- rpart.control(minbucket = 2, minsplit = 4, xval = 10, cp = 0)
fit_clreg <- rpart(diag1 ~ age + gender + cp + trestbps + chol + fbs + 
                         restecg + thalach + exang + oldpeak + slope + ca
                       + thal, data = cleveland, method = "class", 
                       control = model_controls)

# Plot x-value error for model selection
x11()
plot(fit_clreg$cptable[,4], main = "Xval err for model selection", ylab = "cv error")

# Plot Cp for model selection
x11()
plot(fit_clreg$cptable[,1], main = "Cp for model selection", ylab = "Cp Value")
min_cp = which.min(fit_clreg$cptable[,4])

# Prune the tree
pruned_cl <- prune(fit_clreg, cp = 0.05)

# Plot the pruned tree
x11()
plot(pruned_cl, branch = 0.3, compress = T, main = "Pruned Regression Tree")
text(pruned_cl, cex = 0.5)

# Plot the full tree
x11()
plot(fit_clreg, branch = 0.3, compress = T, main = "Full Regression Tree")
text(fit_clreg, cex = 0.5)

# Fit the Logistic Regression Model
log_cl <- glm(diag1 ~ age + gender + cp + trestbps + chol + fbs + restecg + 
                thalach + exang + oldpeak + slope + ca + thal, 
              data = cleveland, family = binomial())
summary(log_cl)