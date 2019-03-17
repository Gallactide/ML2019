import sys, pandas
import spacy
stopwords = list(spacy.load("en").Defaults.stop_words)
from _utilities import *

# Mail Parsing
def parse_mail(mail):
    # This function removes unnecessary headers and splits an email into (header dictionary, body string)
    headers = {}
    headers_r = getHeaders(mail)
    for h,k in headers_r:
        headers[h]=k
    x = mail.split("\n")
    body = [i for i in x if len(i.split(": "))!=2 and len(i)]
    return (headers, stripStopwords("\n".join(body)))
def join_mail(mail):
    # This function joins a header dictionary and body into a regular email string
    return "\n".join([i+": "+mail[0][i] for i in mail[0]])+"\n"+mail[1]
def reduce_mail(mail):
    # This function removes all headers, kinda redundant tbh, used to join mails but decided the headers don't contain enough useful data except maybe the Subject line
    return parse_mail(mail)[1]#join_mail()
def getHeaders(m, whitelist=["Subject", "From", "To", "Date", "X-To", "X-From"]):
    # This function gets the headers in a mailstring, but only returns those that are in the whitelist
    lines = m.split("\n")
    return [i.split(": ") for i in lines if len(i.split(": "))==2 and i.split(": ")[0] in whitelist]
def stripStopwords(text):
    for word in stopwords:
        text = text.replace(word,"")
    return text

@timer
@print_state(text_before_f=lambda f, *args, **kwargs: "[+] Loading & Processing Mails from "+args[0].split("/")[-1]+"...")
def load(path):
    # Loads the emails from a CSV at path, replaces the message column with a parsed version of each message
    mails = pandas.read_csv(path)
    mails["message"] = [reduce_mail(i) for i in mails["message"]]
    return mails
