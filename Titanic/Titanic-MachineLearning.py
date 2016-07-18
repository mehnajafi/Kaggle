
# coding: utf-8
# Simple classification
# In[60]:

import csv as csv
import numpy as np
#read and store data
def read_data(file):
    data = []
    with open(file, 'rt', encoding = "utf8") as csvfile:
        csv_reader = csv.reader(csvfile)

        # This skips the first row of the CSV file.
        # csvreader.next() also works in Python 2.
        header = next(csv_reader)
    
        for row in csv_reader:
            data.append(row[0::])
            
    data = np.asarray(data)
    
    return data


# In[61]:

import numpy as np
def analyze_data(data):
    women_data = data[0::, 4] == 'female'
    men_data = data[0::, 4] == 'male'
    
    women_survived = data[women_data, 1].astype(np.float) == np.float(1)
    men_survived = data[men_data, 1].astype(np.float) == np.float(1)
    
    portion_women_survived = np.sum(women_survived) / np.sum(women_data)
    portion_men_survived = np.sum(men_survived) / np.sum(men_data)
    
    print ("portion of women that survived is %s." % portion_women_survived)
    print ("portion of men that survived is %s" % portion_men_survived)
    
    women_greater = (portion_women_survived > portion_men_survived)
    
    return women_greater


# In[ ]:




# In[67]:

def test(test_file, women_greater):
    test_data = read_data(test_file)
    
    predictions_file = open("gendermodel.csv", "wt")
    csv_writer = csv.writer(predictions_file)
    csv_writer.writerow(["PassengerId", "Survived"])
    
    for row in test_data:
        if(row[3] == "female"):
            if(women_greater):
                csv_writer.writerow([row[0], "1"])
            else:
                csv_writer.writerow([row[0], "0"])
        else:
            if(women_greater):
                csv_writer.writerow([row[0], "0"])
            else:
                csv_writer.writerow([row[0], "1"])
        
    predictions_file.close()


# In[68]:

def titanic_ml(train_file, test_file):
    train_data = read_data(train_file) 

    women_greater = analyze_data(train_data)
    
    test(test_file, women_greater)
    
titanic_ml('/Users/anahita/Desktop/train.csv', '/Users/anahita/Desktop/test.csv')


# In[ ]:



