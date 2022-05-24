# Credit-Card-Fraud-Detection

This project explores a set of credit card customers belonging to a bank, where each customer has the following variables:  
- fraud: an indicator variable for whether the account information has been stolen/compromised (fraud = 1 for fraud and 0 otherwise)
- gender: an indicator variable for gender; the bank only records male or female gender, and the variable is coded as gender = 0 for male and gender = 1 for female
- age: the account holder's age (in years)
- college: an indicator variable for whether the account holder has obtained a bachelor's degree or higher (college = 1 for bachelor's or higher and 0 otherwise)
- score: the account holder's credit score (on a scale of 300 to 850, where higher values mean better creditworthiness)
- amount: the average amount (in U.S. dollars) of the 5 most recent (attempted) charges
- declines: the count of how many out of the 5 most recent charges were declined

The goal is to come up with a way of deciding which credit card accounts with a recent declined charge (at least one decline in the past 5 charges) have been compromised 
by fraudsters and thus should be suspended. The goal is to predict whether or not a credit card account has been compromised by fraud as accurately as possible (measured 
by overall accuracy and AUC values).  

To accomplish this, I used 10-fold cross validation with 6 different classification methods: logistic regression, LDA, GAM, decision trees, random forests, and SVM. 
Within the report, I then dive into simulating the comapany's expected monetary loss using different assumptions and threshold values as described in the paper. The last 
portion of the report explores a hypothetical scenario where the boss of the company would like to check the predictions on a new set of data called new_obs.  

# Files included: 
- R Code 
- PDF that contains analysis of data and code results
- RData file  that contains data
