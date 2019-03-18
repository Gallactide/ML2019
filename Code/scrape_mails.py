import sys, json, csv

def getHeaders(m, whitelist=["Subject", "From", "To", "Date"]):
    lines = m.split("\n")
    return [i.split(": ") for i in lines if len(i.split(": "))==2 and i.split(": ")[0] in whitelist]

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

# replies = {}
replies = []
originals = []
request_emails = 0

with open(sys.argv[1], "r") as f:
    file_paths = [i.replace("\n", "") for i in f.readlines()]

c = 1
r = 0
for path in file_paths:
    try:
        with open(path, "r", errors="ignore") as file:
            mail = file.read().split("-----Original Message-----")
            reply = reduce_mail(mail[0])
            if len(mail)>1:
                replies.append([r,reply])
                for m in mail[1:]:
                    originals.append([path, r, reduce_mail(m.strip().replace("\n>",""))])
                r+=1
            request_emails+=len(mail)-1
        print("[+] Processing",c,"out of",len(file_paths),"mails ")
        sys.stdout.write("\033[F");
        c+=1
    except:
        pass

print("\n[=] Writing",request_emails,"results to", sys.argv[2])
with open(sys.argv[2],"w") as o:
    o_writer = csv.writer(o)
    o_writer.writerow(["path","msg_id","message","needs_reply"])
    for i in originals:
        i.append(1)
        o_writer.writerow(i)
    for i in replies:
        i.append(0)
        i = [-1]+i
        o_writer.writerow(i)
# with open(sys.argv[2].replace(".csv",".replies.csv"),"w") as o:
#     o_writer = csv.writer(o)
#     o_writer.writerow(["msg_id","message","needs_reply"])
#     for i in replies:
#         i.append(0)
#         o_writer.writerow(i)
