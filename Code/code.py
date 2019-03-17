import pandas as pd
import  sklearn
import matplotlib.pyplot as plt
import email, sys
import numpy as np

filename = sys.argv[1]
data = pd.read_csv(filename)

data=pd.read_csv(filename)

data['message'][0]
data.shape

print(data.loc[0]["message"])

def insert_value(dictionary, key, value):
    if key in dictionary:
        values = dictionary.get(key)
        values.append(value)
        dictionary[key] = values
    else:
        dictionary[key] = [value]
    return dictionary

def get_headers(df, header_names):
    headers = {}
    messages = df["message"]
    for message in messages:
        e = email.message_from_string(message)
        for item in header_names:
            header = e.get(item)
            insert_value(dictionary=headers, key=item, value=header)
    print("Successfully retrieved header information!")
    return headers
header_names=["Date", "Subject", "X-Folder", "X-From", "X-To"]
headers = get_headers(data, header_names)

def get_messages(df):
    messages = []
    for item in df["message"]:
        e = email.message_from_string(item)
        message_body = e.get_payload()
        message_body = message_body.lower()
        messages.append(message_body)
    print("Successfully retrieved message body from e-mails!")
    return messages
msg_body = get_messages(data)

data["Message-Body"] = msg_body

#Get employee names
x_from = pd.DataFrame(headers["X-From"], columns=["X-From"])
print(x_from.iloc[:1000]["X-From"].unique()[:10])

if "X-From" not in data.columns:
    data = pd.concat([data, x_from], axis=1, join='inner')

def add_name(df, column, labels):
    new_data = {}
    for item in df[column]:
        tokens = item.split('/')
        for i in range(0, len(labels)):
            value = tokens[i]
            key = labels[i]
            new_data = insert_value(new_data, key, value)
    for key, value in new_data.items():
        df[key] = pd.Series(value)
    print("Successfully added new column!")
    return df
data = add_name(df = data, column="file", labels = ["employee"])

def add_headers(df, header_list):
    for label in header_list:
        df_new = pd.DataFrame(headers[label], columns=[label])
        if label not in df.columns:
            df = pd.concat([df, df_new], axis = 1)
    return df
remaining_headers=["Date", "Subject", "X-To", "X-From", "X-Folder"]
data = add_headers(df = data, header_list = remaining_headers)
print("Here is the emails dataframe after appending all the relevant headers")
print((data.iloc[:1]))

#A sample of randomly selected folders from a newly created folders dataframe shows that there were multiple
# unique folders used by employees.
print(data["X-Folder"].sample(7))

#VISUALISE FOLDER BY SIZE
import seaborn as sns

# dataframe containing counts of every word in the emails dataframe
email_count = data["X-Folder"].value_counts()
indices = email_count.index
count = pd.DataFrame(email_count, columns=["X-Folder"])
count["Folder Names"] = indices
#print(count.head())
def barplot(df, X, Y, figsize, color, orient, ylabel, xlabel, font_scale, rotation):
    f, ax = plt.subplots(figsize=figsize)
    sns.set_color_codes("muted")
    sns.barplot(x=X, y=Y, data=df, color=color, orient=orient)
    ax.set(ylabel=ylabel, xlabel=xlabel)
    sns.set(font_scale=font_scale)
    plt.xticks(rotation=rotation)
    plt.show()

# Figure 1: Bar plot showing 40 folders that contain the most e-mails
barplot(df=count[:40], X="X-Folder", Y="Folder Names", figsize=(7, 8), color='b', orient='h', ylabel="Folders",
        xlabel="Count", font_scale=1.2, rotation=90);

williams = data[data["employee"] == "williams-w3"]
williams["X-Folder"].value_counts()[:8]

# Visualize e-mail count by employee
mail_count = data["employee"].value_counts()
indices = mail_count.index
count = pd.DataFrame(mail_count)
count.rename(columns = {"employee": "Count"}, inplace=True)
count["Employees"] = indices
barplot(df=count[:20], X="Count", Y= "Employees", figsize=(6, 6), color='b', orient='h', ylabel="Employees", xlabel="Count", font_scale=.8, rotation=90);

# Convert date column to datetime
data["Date"] = pd.to_datetime(data["Date"])
print(data.iloc[:1]["Date"])

### Remove Non-topical Folders

def preprocess_folder(datam):
    folders = []
    for item in datam:
        if item is None or item is '':
            folders.append(np.nan)
        else:
            item = item.split("\\")[-1]
            item = item.lower()
            folders.append(item)
    print("Folder cleaned!")
    return folders
data["X-Folder"] = preprocess_folder(data["X-Folder"])

# Folders we can filter out
unwanted_folders = ["all documents", "deleted items", "discussion threads", "deleted Items" "sent", "inbox", "sent items", "sent mail", "untitled", "notes inbox",
                    "junk file", "calendar"]

# A new dataframe without non-topical folders
data = data.loc[~data['X-Folder'].isin(unwanted_folders)]
print(data.iloc[:15]["X-Folder"].value_counts())

#Extract e-mails for employees who had over 2000 e-mails
email_count = dict(data["employee"].value_counts())
reduced_emails = [key for key, val in email_count.items() if val >= 2000]
data = data.loc[data['employee'].isin(reduced_emails)]
print(data["employee"].value_counts())

data.isnull().sum()
#remove rows with missing values
data.dropna(inplace=True)

#tokenization
def tokenize(row):
    if row is None or row is '':
        tokens = ""
    else:
        tokens = row.split(" ")
    return tokens

#remove regural expressions
import re
def remove_reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', " ", token)
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens

#stop words removal-NLTK library
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token

def assemble_bag(datam):
    datam = datam.apply(tokenize)
    datam = datam.apply(stop_word_removal)
    datam = datam.apply(remove_reg_expressions)

    unique_tokens = []
    single_tokens = []

    for item in datam:
        for token in item:
            if token in single_tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)
            else:
                single_tokens.append(token)

    df = pd.DataFrame(0, index=np.arange(len(datam)), columns=unique_tokens)

    for i, item in enumerate(datam):
        for token in item:
            if token in unique_tokens:
                df.iloc[i][token] += 1
    return df


#focus on just one Enron employee for our classification problem
employee = data[data["employee"] == "kaminski-v"]

def remove_folders(datam, n):
    # Returns the folders containing more than 'n' number of e-mails
    email_count = dict(data["X-Folder"].value_counts())
    small_folders = [key for key, val in email_count.items() if val <= n]
    datam = datam.loc[~datam['X-Folder'].isin(small_folders)]
    return datam

n=200
employee = remove_folders(employee, n)

#Encoding Class Labels
from sklearn.preprocessing import LabelEncoder

def label_encoder(df):
        class_le = LabelEncoder()
        y = class_le.fit_transform(df['X-Folder'])
        df.loc[:, 'y'] = y
        return df

label_encoder(employee)
print("Randomly selected labels representing our folders")
unique_folders = employee["y"].unique()
print(unique_folders)

#Sort data chronologically
def sort(df):
    df = df.sort_values(by="Date", axis=0)
    return df

employee = sort(employee)
print(employee.loc[:,("Date", "y", "X-Folder")][:10])

def prepare_features(df):
    from_bag = assemble_bag(df["X-From"])
    to_bag = assemble_bag(df["X-To"])
    message_bag = assemble_bag(df["Message-Body"])
    subject_bag = assemble_bag(df["Subject"])
    frames = [from_bag, subject_bag, to_bag, message_bag]
    X = pd.concat(frames, axis=1, join='inner')
    return X
 #X.drop(labels=[" ", ""], inplace=True, axis=1)

X = prepare_features(employee)

print(X.ix[:3, 1000:])
print("A peek into some of the columns in the features dataframe")

#TRAINING
from sklearn.linear_model import LogisticRegression

def split_data(X, employee):
    training_size = int(len(employee) * 0.8)
    X_train = X[:training_size]
    X_test = X[training_size:]
    y_test = employee[training_size:]["y"]
    y_train = employee[:training_size]["y"]
    return X_train, X_test, y_test, training_size

X_train, X_test, y_test, training_size = split_data(X, employee)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
X_train_tf = tf_transformer.transform(X_train)
X_train_tf.shape
