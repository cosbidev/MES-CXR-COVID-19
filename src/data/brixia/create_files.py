import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import pandas as pd

task = "brixia"

data_file = os.path.join("./data/processed", task, "data.xlsx")
dest_dir = os.path.join("./data/processed", task)

db = pd.read_excel(data_file, header=0, index_col="img")
label_col = "label"

all = []
# all  
with open(os.path.join(dest_dir, 'all.txt'), 'w') as file:
    file.write("img label\n")
    for img in db.index:
        label = db.loc[img, label_col] + "\n"
        row = img+" "+label
        file.write(row)
        all.append(row)
