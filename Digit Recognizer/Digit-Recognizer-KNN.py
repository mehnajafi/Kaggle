
# coding: utf-8

# In[12]:

from sklearn.neighbors import KNeighborsClassifier
import csv as csv
import numpy as np

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


# In[13]:

def write_data(test_labels):
    predictions_file = open("digit_predictions.csv", "wt")
    csv_writer = csv.writer(predictions_file)
    csv_writer.writerow(["ImageId", "Label"])
    
    image_id = 0
    for label in test_labels:
        csv_writer.writerow([image_id, label])
        image_id = image_id + 1
    
    predictions_file.close()


# In[14]:

def digit_recognizer(train_file, test_file):
    train_data = read_data(train_file)
    
    #get labels and features
    labels = train_data[:, 0]
    features = train_data[:, 1:]
    
    #train using knn
    print('training phase has been started')
    knn = KNeighborsClassifier()
    knn.fit(features, labels)
    print('training phase has been finished')
    
    #test the model
    print('testing phase has been started')
    test_data = read_data(test_file)
    ##test_data = test_data[0:10, :]
    test_labels = knn.predict(test_data)
    print('testing phase has been finished')
    
    #write predictions
    write_data(test_labels)
    
digit_recognizer('/Users/anahita/Documents/PythonRepository/Digit Recognizer/train.csv', '/Users/anahita/Documents/PythonRepository/Digit Recognizer/test.csv')


# In[ ]:



