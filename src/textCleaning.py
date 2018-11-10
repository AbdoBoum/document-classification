import string,re
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import *

stop = set(stopwords.words('french'))
stop1 = open("../dataset/stop-words/list.txt", encoding="utf8").read().splitlines()
stop2 = set(stopwords.words('english'))

def clean_text_en(text) :
    text = [word for word in text.split()
                        if ((word.lower() not in stop) and (len(word.lower()) < 15) and (
                        word.lower() not in stop1)
                        and (word.lower() not in stop2) and not (is_number(word.lower())))]
    text = " ".join(text)
    text = text.strip()
    return text

def clean_text(text) :
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    text = text.lower().split()
    ##Remove stop words
    stops = set(stopwords.words("arabic"))
    text = [w for w in text if not w in stops and len(w) >= 3 and not (is_number(w.lower()))]
    text = " ".join(text)


    ## Clean Text
    text = re.sub('http\S+\s*', '', text)  # remove URLs
    text = re.sub('RT|cc', '', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuations
    text = re.sub('\s+', ' ', text)  # remove extra whitespace

    ## Remove Emojis

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    ## Keep only arabic words
    #text = re.sub('[^.,&&[\\P{InArabic}\\p{P}\\p{Digit}]]+','',text)
    ## Clean the text
    #text = re.sub(r"[A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text)

    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel," ", text)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()
    ## Stemming
    #text = steamText(text)
    return text

def steamText(text):
    text = text.split()
    stemmer = ISRIStemmer()
    stemmed_words = [stemmer.stem(word) for word in text]
    return " ".join(stemmed_words)

def steamTextEn(text):
    text = text.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text]
    return " ".join(stemmed_words)

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
