echo "[@] Downloading Enron dataset..."
# wget -q --show-progress https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz -O enron.tar.gz

echo "[#] Expanding '\033[1menron.tar.gz\033[0m' to /dataset/maildir..."
# mkdir ./dataset/ > /dev/null 2>/dev/null
# cd ./dataset; tar xf ../enron.tar.gz maildir; cd ..

echo "[%] Determining eligible mails..."
grep -lri "subject: re:" dataset/maildir | head -n50 > thread_paths.txt
echo "[*] Running Parser to get augmented csv..."
python code/scrape_mails.py thread_paths.txt dataset/parsed.csv

echo "[-] Cleaning up..."
# rm enron.tar.gz
# rm thread_paths.txt

echo "[$] Parsed mails are now available at \033[1m'./dataset/parsed.csv'\033[0m"
