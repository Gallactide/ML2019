import sys, pandas, sklearn
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

def parse_mail(mail):
    headers = {}
    headers_r = getHeaders(mail)
    for h,k in headers_r:
        headers[h]=k
    x = mail.split("\n")
    body = [i for i in x if len(i.split(": "))!=2 and len(i)]
    return (headers, "\n".join(body))

def join_mail(mail):
    return "\n".join([i+": "+mail[0][i] for i in mail[0]])+"\n"+mail[1]

def reduce_mail(mail):
    return join_mail(parse_mail(mail))

def run_between(text, text_b, function, *args, **kwargs):
    sys.stdout.write(text)
    sys.stdout.flush()
    out = function(*args, **kwargs)
    print(text_b)
    return out

def load(path):
    sys.stdout.write("[+] Loading Mails from CSV...")
    sys.stdout.flush()
    mails = pandas.read_csv(path)
    print(" done.")
    return mails

def getHeaders(m, whitelist=["Subject", "From", "To", "Date"]):
    lines = m.split("\n")
    return [i.split(": ") for i in lines if len(i.split(": "))==2 and i.split(": ")[0] in whitelist]

def vectorize_count(mails):
    vec = CountVectorizer()
    data = vec.fit_transform(mails)
    return (vec, data)

def vectorize_hash(mails):
    vec = HashingVectorizer(n_features=2**8)
    data = vec.fit_transform(mails)
    return (vec, data)

if __name__ == '__main__':
    mails = load(sys.argv[1])
    processed_mails = [reduce_mail(mails["message"][i]) for i in range(0,10)]
    vec, data = run_between("[@] Running Vectorizer...", " completed", vectorize_hash, processed_mails)
