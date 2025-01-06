# importing the dataset from local
import pandas as pd
import numpy as np
emails = pd.read_csv('emails.csv')
# importing dataset from github
'''To commit later'''
# data preprocessing
def process_email(text):

    '''This function does the following to the input string/sentence in the given order
    1. Uses the lower() method to turn all the words into lower case
    2. Uses the split() method to turn the lower case words into a list
    3. Checks whether each word appears in the email. 
    4. We are not really concerned about the number of times it appears so we turn it into a set'''

    lower_case_text = text.lower()  # 1. Convert to lowercase
    words_set = set(lower_case_text.split())
    words_list = list(words_set)
    return words_list


# Apply processing functions on the data
emails['words'] = emails['text'].apply(process_email)
emails.head()

# Finding the prior probablity
p_prior = sum(emails['spam'])/len(emails)

# Finding the posteriors with Bayes's theorem
model = {}
for index, email in emails.iterrows():
    for word in email['words']:
        if word not in model:
            model[word] = {'spam': 1, 'ham': 1}
        if word in model:
            if email['spam']:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

# Implementing the Naive Bayes Algorithm
def predict_naive_bayes(email):
    # 1.Calculates the total number of emails, spam emails and ham emails
    total = len(emails)
    num_spam = sum(emails['spam'])
    num_ham = total - num_spam
    # 2. Processes each email by turning it into list of its words in lowercase
    lower_email = email.lower()
    words = set(email.split())
    spams = [1.0]
    hams = [1.0]
    # 3. For each word computes the conditional probablity that an email containing that word is spam or ham as a ratio
    for word in words:
        if word in model:
            spams.append((model[word]['spam']/num_spam)*total)
            hams.append((model[word]['ham']/num_ham)*total)
    # 4. Multiplies all the previous probablities times the prior probablity of the email being spam, and calls this prod_spams, does a similar process for prod_hames.
    prod_spams = np.long(np.prod(spams)*num_hams)
    prod_hams = np.long(np.prod(hams)*num_ham)
    # 5. Normalises the two probablities to get them to add to one (using Bayes' theoren) and return the result
    return prod_spams/(prod_spams + prod_hams)

