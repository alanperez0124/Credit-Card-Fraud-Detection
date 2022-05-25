# Alan Perez


# Clear environment and set working directory
rm(list=ls())
setwd("C:/Users/ajper/Desktop/STA 4350/Final Submission")


# Load in data
load("cc_fraud.RData")
attach(cc_fraud)
str(cc_fraud)
set.seed(4350)


# Load in packages
library(MASS)
library(gam)
library(tree)
library(pROC)
library(e1071)
library(randomForest)

# Constants
tau <- 0.5

# ----- DEFINE FUNCTIONS -----
# create a funtion to plot ROC cruves: 

# function arguments (inputs to the function): 
# y is the actual true y values (0's or 1's)
# ph is the pi_hat values (i.e., the predicted probabilitites of y being 1 
# for each of the observations)
# clr is the color to use when plotting the curve
# new, where new=TRUE means draw a new plot from scratch, and new=FALSE 
#  means the curve to whatever plot is already open

draw_roc <- function(y, ph, clr="red", new=TRUE){
  taus <- seq(0, 1, 0.001)      # thresholds for ph for classifying y as 1
  sens <- numeric(length(taus)) # vector to store sensitivity values
  fpr <- numeric(length(taus))  # vector to store false positive rate vals
  for (i in 1 : length(taus)) {
    sens[i] <- mean(ph[y==1] > taus[i])
    fpr[i] <- mean(ph[y==0] > taus[i])
  }
  
  # if new==TRUE, draw a new plot: 
  if (new) {
    plot(NA, NA, type="n", xlim=c(0, 1), ylim=c(0, 1), 
         xlab="False positive rate", 
         ylab="Sensitivity")
    abline(h=c(0, 1), v=c(0, 1), a = 0, b = 1)
  }
  lines(fpr, sens, lwd=2, col=clr)
}


# Create a function to calculate the area under the ROC curve or AUC for short: 

# Function arguments (inputs): 
# y is the actual true y values (0's or 1's)
# ph is the pi_hat values (i.e., the predicted probabilities of y being 1 for 
#  each of the observations)

# Return value (output): 
#  AUC value 
calc_auc <- function(y, ph) {
  taus <- seq(0, 1, 0.001)      # thresholds for ph for classifying y as 1
  sens <- numeric(length(taus)) # vector to store sensitivity values
  fpr <- numeric(length(taus))  # vector to store false positive rate vals
  for (i in 1 : length(taus)) {
    sens[i] <- mean(ph[y==1] > taus[i])
    fpr[i] <- mean(ph[y==0] > taus[i])
  }
  sens <- c(1, sens)
  fpr <- c(1, fpr)
  # note that negative indexing removes the value at that index
  sum( (sens[-1] + sens[-length(sens)])/2 * (fpr[-length(fpr)] - fpr[-1]))
} 


# ----- Create training and testing data sets -----
#set.seed(4350)
#test_set <- sort(sample(nrow(cc_fraud), nrow(cc_fraud)/2))
#train_set <- (1:nrow(cc_fraud))[-test_set]  



# ---- K-fold -----
n <- nrow(cc_fraud)
k <- 10
fold_size <- n/k

# We will make a matrix where first col is first fold, 2nd col is 2nd fold, etc
folds <- matrix(sample(n, n, replace=FALSE), fold_size, k)

# ----- SUMMARY STATISTICS -----
# Fraud
table(fraud)/length(fraud)

# Gender
table(gender)/length(gender)

# Age
summary(age)
boxplot(age, main="Age")

# College
table(college)/length(college)

# Score
summary(score)
boxplot(score, main="Score")

# Amount
summary(amount)
boxplot(amount, main="Amount")

# Declines
summary(declines)
boxplot(declines, main="Declines")


# ----- PREDICTIVE METHODS -----
# --------------------LOGISTIC REGRESSION-----------------------------------
# 
"Try exploring a quadratic term or soemthing "
tau <- 0.5
pi_hat_glm <- numeric(n)
for (i in seq_len(k)){
  # Fit the logistic model exlcuding observations in fold i: 
  my_glm <- glm(fraud ~ gender + age + college + score + amount + declines, 
                family=binomial, subset=-folds[, i])
  
  # fill in the phat values for observations in fold i: 
  pi_hat_glm[folds[, i]] <- predict(my_glm, type="response", newdata=cc_fraud[folds[, i], ])
}

# Create y_hat for he glm 
yhat_glm <- as.numeric(pi_hat_glm > tau)

# Overall accuracy
mean(fraud== yhat_glm)

# Calculate AUC
logistic_auc <- calc_auc(fraud, pi_hat_glm)
logistic_auc
auc(roc(fraud, pi_hat_glm))


# ROC curve
draw_roc(fraud, pi_hat_glm)

# --------------------LINEAR DISCRIMINANT ANALYSIS----------------------------
# 
pi_hat_lda <- numeric(n)
for (i in seq_len(k)) {
  # Fit the LDA exlcusing observations in fold i: 
  my_lda <- lda(fraud ~ gender + age + college + score + amount + declines, 
                subset=-folds[, i])
  
  # Fill in phat values for observations in fold i: 
  pi_hat_lda[folds[, i]] <- predict(my_lda, newdata=cc_fraud[folds[, i], ])$posterior[,2]
}

# Create Y_hat for lda
yhat_lda <- as.numeric(pi_hat_lda > tau)

# Overall Accuracy
mean(fraud == yhat_lda)

# Calculate AUC 
lda_auc <- calc_auc(fraud, pi_hat_lda)
lda_auc
# auc(roc(fraud, pi_hat_lda)) # Check answer

# ROC Curve
draw_roc(fraud, pi_hat_lda)
# plot(roc(fraud, pi_hat_lda)) # Check answer

# --------------------GENERALIZED ADDITIVE MODEL-------------------------------
#  
pi_hat_gam <- numeric(n)

for (i in seq_len(k)) {
  # FIt the GAM excluding observations in fold i: 
  my_gam <- gam(fraud ~ gender + s(age, 4) + college + s(score, 4) + s(amount, 4) + s(declines, 4), 
                subset=-folds[, i], family=binomial )
  
  # Fill in phat values for observations in fold i: 
  pi_hat_gam[folds[, i]] <- predict(my_gam, newdata=cc_fraud[folds[, i], ], type="response")
}


# Create y_hat for gam
yhat_gam <- as.numeric(pi_hat_gam > tau)

# Overall accuracy
mean(fraud == yhat_gam)


# Calculate AUC
gam_auc <- calc_auc(fraud, pi_hat_gam)
gam_auc
# auc(roc(fraud, pi_hat_gam)) # Check answer

# ROC Curve
draw_roc(fraud, pi_hat_gam)
# plot(roc(fraud, pi_hat_gam)) # Check answer

# --------------------DECISION TREES--------------------------------
# 
fraud_factor <- factor(fraud)
pi_hat_tree <- numeric(n)

for (i in seq_len(k)) {
  # Fit the tree excluding observations in fold i: 
  my_tree <- tree(fraud_factor ~ gender + age + college + score + amount + declines, 
                  subset=-folds[, i] )
  
  # Fill in the phat values for observations in fold i: 
  pi_hat_tree[folds[, i]] <- predict(my_tree, newdata=cc_fraud[folds[, i], ])[, 2]
}

# Create y_hat for trees
yhat_tree <- as.numeric(pi_hat_tree > tau)

# Overall accuracy
mean(fraud == yhat_tree)

# Calculate AUC
tree_auc <- calc_auc(fraud, pi_hat_tree)
tree_auc
# auc(roc(fraud, pi_hat_tree)) # check answer

# ROC curve
draw_roc(fraud, pi_hat_tree)
# plot(roc(fraud, pi_hat_tree))

# --------------------RANDOM FORESTS---------------------------------
# 
fraud_factor <- factor(fraud)
set.seed(4350)
pi_hat_forest <- numeric(n)

for (i in seq_len(k)) { 
  # Fit the random forest excluding observations in fold i: 
  my_forest <- randomForest(fraud_factor ~ gender + age + college + score + 
                              amount + declines, subset=-folds[, i] )
  
  
  # Fill in teh phat values for observations in fold i: 
  pi_hat_forest[folds[, i]] <- predict(my_forest, type="prob", 
                                       newdata=cc_fraud[folds[, i], ])[, 2]
}

# Create y_hat for forest
yhat_forest <- as.numeric(pi_hat_forest > tau)

# Overall accuracy
mean(fraud == yhat_forest)

# Calculate AUC
forest_auc <- calc_auc(fraud, pi_hat_forest)
forest_auc
auc(roc(fraud, pi_hat_forest))                          

# Draw ROC
draw_roc(fraud, pi_hat_forest)
plot(roc(fraud, pi_hat_forest))

# --------------------CORRECT SVM --------------------
my_data <- data.frame( fraud, gender, age, college, score, amount, declines )
pi_hat_svm <- numeric(n)

for (i in seq_len(k)) {
  # Fit the random forest excluding observations in fold i: 
  my_svm <- svm(fraud ~ gender + age + college + score + amount + declines, 
                data=my_data, type="C", probability=TRUE, subset=-folds[, i])
  
  # Fill in the p_hat values for observations in fold i:
  svm_predictions <- predict(my_svm, newdata=cc_fraud[folds[, i], ], probability=TRUE)
  
  pi_hat_svm[folds[, i]] <- attributes(svm_predictions)$probabilities[, 2]
}

# Create Y_hat for svm
yhat_svm <- as.numeric(pi_hat_svm > tau)

# Overall accuracy
mean(fraud == yhat_svm)

# Calculate AUC
svm_auc <- calc_auc(fraud, pi_hat_svm)
svm_auc
auc(roc(fraud, pi_hat_svm))

# ROC curve
draw_roc(fraud, pi_hat_svm)
plot(roc(fraud, pi_hat_svm))



# ----- GOAL 2 ----
# GOAL 2: FIND WAY TO PREDICT WHETHER OR NOT A CREDIT CARD ACCOUNT HAS BEEN 
#         COMPROMISED THAT MINIMIZES THE COMPANY'S EXPECTED MONETARY LOSS
# Find the best model 
logistic_auc
lda_auc
gam_auc
tree_auc
forest_auc
svm_auc
"Best model seems to be the Generalized additive model"

calc_loss <- function(y, yhat, costs) {
  # Look at all combinations of actual and predicted values
  combinations <- matrix(rep(0, times=4), ncol=2, byrow=TRUE) # Create empty table
  colnames(combinations) <- c(0, 1) # yhat 
  rownames(combinations) <- c(0, 1) # fraud
  combinations <- as.table(combinations)
  
  combinations[1, 1] <- sum(fraud[fraud==0] == yhat[fraud==0])# all the times yhat = fraud == 0
  combinations[2, 2] <- sum(fraud[fraud==1] == yhat[fraud==1])# all times yhat=fraud = 1
  
  combinations[2, 1] <- sum(fraud[fraud==1] - yhat[fraud==1]) # y = 1, but yhat = 0
  combinations[1, 2] <- sum(yhat[fraud==0] - fraud[fraud==0]) # y = 0, but yhat = 1
  
  # If Y = 0 (not fraudulent), but Y_hat = 1 => loss = $25 (false positive)
  # If Y = 1 (fraud), but Y_hat = 0 => loss = $430.33      (false negatives)
  return(combinations[1, 2]*25 + combinations[2, 1]*costs)
  #return(total)
}

# CASE 1: Company will pay sample mean transaction of $430.33
# First we find the optimal tau value
taus <- seq(0, 1, 0.01)
costs <- numeric(length(taus))
for (i in 1 : length(taus)) {
  yhat <- as.numeric(pi_hat_gam > taus[i])
  
  cost <- calc_loss(fraud, yhat, 430.33) / 1000
  costs[i] <- cost
}


plot(taus, costs, pch=20, ylab="Cost per Customer")

# Set tau to 0
tau <- 0

# Find the y_hat values using the new tau
yhat <- as.numeric(pi_hat_gam > tau)
yhat

result <- calc_loss(fraud, yhat, 430.33)
result / 1000

#
#
#
#
#

# CASE 2: Cost varies from customer to customer
# First find optimal tau value
taus <- seq(0, 1, 0.01)
costs <- numeric(length(taus))
for (i in 1 : length(taus)) {
  yhat <- as.numeric(pi_hat_gam > taus[i])
  
  
  total <- 0
  
  for (j in 1 : length(amount)){
    # If we predict not fraudulent but is in fact fraudulent
    if ((yhat[j] == 0) && (fraud[j] == 1)) {
      costs[i] <- costs[i] + amount[j]
    }
    # If we predict fraudulent but is not fraudulent
    else if ((yhat[j] == 1) && (fraud[j] == 0)) {
      costs[i] <- costs[i] + 25
    }
  }
}

plot(taus, costs/1000, pch=20, ylab="Cost per Customer")

# Set tau to 0
tau <- 0

yhat <- as.numeric(pi_hat_gam > tau)
total <- 0

for (i in 1 : length(amount)){
  # If we predict not fraudulent but is in fact fraudulent
  if ((yhat[i] == 0) && (fraud[i] == 1)) {
    total <- total + amount[i]
  }
  # If we predict fraudulent but is not fraudulent
  else if ((yhat[i] == 1) && (fraud[i] == 0)) {
    total <- total + 25
  }
  
}

total/1000


# ----- CODE SNIPPET-----
set.seed(4350)
new_obs <- cc_fraud[1:10, ]

# K-fold 
n <- nrow(cc_fraud)
k <- 10
fold_size <- n/k

# We will make a matrix where first col is first fold, 2nd col is 2nd fold, etc
folds <- matrix(sample(n, n, replace=FALSE), fold_size, k)

# Get the pi_hat values
pi_hat_gam <- numeric(n)

for (i in seq_len(k)) {
  # FIt the GAM excluding observations in fold i: 
  my_gam <- gam(fraud ~ gender + s(age, 4) + college + s(score, 4) + s(amount, 4) + s(declines, 4), 
                subset=-folds[, i] )
  
  # Fill in phat values for observations in fold i: 
  pi_hat_gam[folds[, i]] <- predict(my_gam, newdata=cc_fraud[folds[, i], ])
}
pi_hat_new_obs <- predict(my_gam, newdata=new_obs)
yhat <- as.numeric(pi_hat_new_obs > 0)
yhat





















