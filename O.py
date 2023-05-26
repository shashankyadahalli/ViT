import os
import csv

import pandas as pd
folder_path = 'D:/VisionTransformer/Test'

# Path to the CSV file containing the file IDs
csv_path = 'D:/VisionTransformer/sample_submission.csv'

# Create a set of file IDs from the CSV file
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    file_ids = list([row[0]+".tif", row[1]]for row in csv_reader)

# Get a list of file names in the folder
file_names = os.listdir(folder_path)


for i in (file_ids):
    if i[0] in file_names:
        print(i[0].split(".")[0]+","+i[1])
# Compare file names with file IDs
