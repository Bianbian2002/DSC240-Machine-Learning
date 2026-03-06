# Credit Card Approval Prediction

Credit score cards are a common risk control method in the financial industry. It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank is able to decide whether to issue a credit card to the applicant. Credit scores can objectively quantify the magnitude of risk.

Generally speaking, credit score cards are based on historical data. Once encountering large economic fluctuations. Past models may lose their original predictive power. Logistic model is a common method for credit scoring. Because Logistic is suitable for binary classification tasks and can calculate the coefficients of each feature. In order to facilitate understanding and operation, the score card will multiply the logistic regression coefficient by a certain value (such as 100) and round it.

At present, with the development of machine learning algorithms. More predictive methods such as Boosting, Random Forest, and Support Vector Machines have been introduced into credit card scoring. However, these methods often do not have good transparency. It may be difficult to provide customers and regulators with a reason for rejection or acceptance.


## Task

Given the client’s historical behaviors, the bank can identify whether this client is a risky customer that tends to pay late billing. Combining this information with the corresponding application, the bank can make a better decision on credit card application to reduce the number of risky customers. Usually, users in risk should be 3%, so the imbalance data is a big problem. 

In this project, we already identify the risky customers (target=1) and the non-risky customers (target=0) based on the historical data. Specifically, the customers with target=1 are the customers who have at least one overdue payment for more than 60 days. We also modify the data to make the risky ratio at around 10%.

The goal is to build a model to predict whether the application should be approved (non-risky customer, target=0) or not (risty customer, target=1).


## Data format
| Feature Name        | Meaning                  | Feature Type             |
| ------------------- | ------------------------ |  ----------------------- |  
| CODE_GENDER         | Gender                   | binary (F/M)             |
| FLAG_OWN_CAR        | Is there a car           | binary (Y/N)             |
| FLAG_OWN_REALTY     | Is there a property      | binary (Y/N)             | 
| CNT_CHILDREN        | Number of children       | categorical              |
| AMT_INCOME_TOTAL    | Annual income            | continuous               |
| NAME_INCOME_TYPE    | Income category          | categorical              | 
| NAME_EDUCATION_TYPE | Education level          | categorical              |  
| NAME_FAMILY_STATUS  | Marital status           | categorical              | 
| NAME_HOUSING_TYPE   | Way of living            | categorical              | 
| DAYS_BIRTH          | Birthday                 | continuous               |
| DAYS_EMPLOYED       | Start date of employment | continuous               |
| FLAG_MOBIL          | Is there a mobile phone  | binary (0/1)             |
| FLAG_WORK_PHONE     | Is there a work phone    | binary (0/1)             |
| FLAG_PHONE          | Is there a phone         | binary (0/1)             |
| FLAG_EMAIL          | Is there an email        | binary (0/1)             |
| OCCUPATION_TYPE     | Occupation               | categorical              |
| CNT_FAM_MEMBERS     | Family size              | categorical              |
| QUANTIZED_INC       | quantized income         | categorical              |
| QUANTIZED_AGE       | quantized age            | categorical              |
| QUANTIZED_WORK_YEAR | quantized employment year| categorical              |
| target              | whether the user is a risky customer | binary (0/1)             |


Explanation:
- DAYS_BIRTH: Count backwards from current day (0), -1 means yesterday
- DAYS_EMPLOYED: Count backwards from current day(0). If positive, it means the person currently unemployed.
