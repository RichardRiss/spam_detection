import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import multiprocessing
import seaborn as sns
import email
import os


df = pd.read_csv("emails.csv")
f = open('enron.csv', 'w')

for mail in df.loc:
    e = email.message_from_string(mail['message'])
    hdate = e.get('Date')
    hfrom = e.get('From')
    hsubject = e.get("Subject")
    content = e.get_payload()
    #f.write("\"b\'Date: {} From: {} Subject: {} {} \", 1\n".format(hdate,hfrom,hsubject,content.replace('\n', ' ')))
    content = content.replace('\n', ' ')
    content = content.replace("\"", '')
    f.write("\"b\'{}\'\", 1\n".format(content))

f.close()