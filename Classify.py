#include necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier #Random Forest modal
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
#loading dataset
data=pd.read_csv('spam.csv')
data.head()
#May also include other data handling metods 

x_train=data['Message']+" "+data['Category'] #traning data preparation (Message & Category are col contains email and its Category i.e spam or not

#vectorizing training data
vectorizer=HashingVectorizer()
x_trained=vectorizer.fit_transform(x_train)
print(x_trained)#optional

#Converting Categorical value into numerical value & storing it in col named spam
data['spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
y_train=data['spam']

#Traning Modal
rf_classifier=RandomForestClassifier(n_estimators=100,random_state=42)
rf_classifier.fit(x_trained,y_train)

#Making predictions
new_input_data = [
   """Hello,

I hope this email finds you well. I wanted to reach out to inform you about an upcoming event that may be of interest to you.

Our company is hosting a charity fundraiser next month to support [charity name or cause]. We're excited to bring our community together for a great cause and would love for you to join us!

Please find more information about the event in the attached brochure.

Thank you for your consideration. We hope to see you there!

Best regards,
[Your Name]
[Your Position]
[Your Company]
"""
] #email wants to be predicted

#Vectoroizing new_input-data
new_input_vectorized = vectorizer.transform(new_input_data)
print(new_input_vectorized) #Optional

# Making prediction
predictions = rf_classifier.predict(new_input_vectorized)
if predictions==1: print("SPAM")
else: print("NOT SPAM")



