import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csv_file:
        # create csv handler
        csv_reader = csv.reader(csv_file)

        # skip the column label row
        next(csv_reader)

        for row in csv_reader:
            # get the month
            dates.append(int(row[0].split('-')[2]))
            prices.append(float(row[1]))


