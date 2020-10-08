# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:23:24 2020

@author: Chandramouli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import seaborn as sns
import nltk
import numpy as np
import re
import tqdm
import unicodedata
#import contractions
import spacy
from nltk.stem import WordNetLemmatizer,PorterStemmer
import en_core_web_sm
ps =PorterStemmer()
lemmatizer = WordNetLemmatizer()

nlp = en_core_web_sm.load()
ps = nltk.porter.PorterStemmer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')


test=pd.read_excel('NLP Engineer -Test Dataset.xlsx')
train=pd.read_excel('NLP Engineer -Train&val Dataset.xlsx')

train.info()
test.info()
#filling NA
train['Conversations']=train['Conversations'].fillna('')
test['Conversations']=test['Conversations'].fillna('')

#PREPROCESSING
'''
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):#diacritic
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


import tqdm


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def spacy_lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text])
    return text
#filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
#stem_words=[stemmer.stem(w) for w in filtered_words]
#lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

def simple_stemming(text, stemmer=ps):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = re.sub(r'((http|ftp|https|www):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', doc, flags=re.MULTILINE)

        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = expand_contractions(doc)
        doc=remove_stopwords(doc)
        doc=spacy_lemmatize_text(doc)
        doc=simple_stemming(doc)
        
       
  #      doc=remove_punctuation(doc)

        
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        
        #doc = re.sub(r"http\S+", "", doc)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs
#filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
#stem_words=[stemmer.stem(w) for w in filtered_words]
#lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
train.info()
train['text']= pre_process_corpus(train['Conversations'])
train=train.drop(['Conversations'], axis=1)'''


x1=pd.DataFrame(x)
x1.columns=['text']
'''text = "Originally Posted by Birddog Cause of death was an ""enlarged heart"". I'm pretty sure Abe has repeatedly said that this is easily detectable and these types of incidents could be avoided. Very sad incident. Not a physician but I have heard contradictory opinions regarding the benefits of wide scale testing: http://www.sca-aware.org/campus/camp...-heart-disease http://news.heart.org/screening-youn...heart-disease/ I have always thought that Ronny so blessed to have his condition detected and fixed. 999990019 25-Jul-2016 http://guboards.spokesmanreview.com/showthread.php?57567-Oklahoma-State-Basketball-player-dead-after-workout#post1234503"
text = re.sub(r'https?:\/\/\S*', '', text, flags=re.MULTILINE)'''


import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


import tqdm


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    sentence=expand_contractions(sentence)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    #rem_url=re.sub(r'http\S+', '',cleantext)
    rem_url = re.sub(r'((http|ftp|https|www):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', cleantext, flags=re.MULTILINE)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')#[^a-zA-Z0-9_]
    tokens = tokenizer.tokenize(rem_num)  
    
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

train1=train
train['text']= train['Conversations'].map(lambda s:preprocess(s)) 
train1['text']= train1['Conversations'].map(lambda s:preprocess(s)) 

train.info()
train=train.drop(['Conversations'],axis=1)

#POS TAG
train1=pd.read_excel('NLP Engineer -Train&val Dataset.xlsx')


from nltk.tokenize import sent_tokenize 



sentences = [sent_tokenize(document) for document in train1['Conversations']]
nltk.download('averaged_perceptron_tagger')
sentences1 = [nltk.pos_tag(sent) for sent in sentences]

#Extracting email ID
def extract_email_addresses(text):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(text)


email=extract_email_addresses('WEEKEND LIGHTER: FEEL HUNGER (February 27/28, 2016, No. 9/2016) Feel free to mail your views on this edition of WL to mgwarrier@gmail.com I FIRST PERSON Another view on inequality and hunger On the morning of February 24, 2016 a small shop near Bhandup Railway Station caught fire and was gutted. I couldn?t go to the newspaper vender as smoke and fire force scared me. I missed my usual finance newspapers that morning. Any calamity is an opportunity. That gave me a chance to read ?The Rhetoric Of Inequality? by former Procter & Gamble CEO Gurucharan Das published in The Times of India that day. The article gives a fair idea about how those who have never been hungry, or have forgotten the ?hungry days? looks at hunger. Das claims that ?carping over inequality? tantamount to ?debating what was settled long ago- you don?t make the poor rich by making the rich poor?. Excerpts from the article: ?I have always believed that it is none of my business how much the Ambanis earn as long as they create lots of jobs, pay their taxes and produce wealth for the society. The aam aadmi cares mostly about how he is fairing; he sometimes compares himself to his friends but never to the filthy rich. Judging the lifestyle of others tempts one to control other things, and this is a short step to becoming a command society. Not to live ostentatiously is a call of Dharma, not a legal duty.? I am totally ?brain-washed? by Gurucharan Das and I leave it to you think for yourself, for now. If I am able to come out of the ?damage?, I will be back with my views later! II FROM HERE AND THERE February 22, 2016 Needed, higher insurance cover This refers to Aarati Krishnan?s article ?Deposit insurance: what you didn?t know? (Real returns, February 22). The article brings together in one place several FAQs on deposit insurance with answers explaining factual position. The article will serve a larger purpose, if policy makers take cognizance of the constraints within which DICGC is functioning today and initiate moves that will help the organisation recast its vision and mission to conform to the present scenario in which banks and financial institutions are working in India today. Such a makeover for DICGC may have to factor in: (i) The resources at DICGC?s command at Rs50,000 crore is not small. But, it is inadequate to meet the business expansion the corporation may have to think of, if the confidence deposit insurance should instil among depositors is to be restored. DICGC may also have to revisit credit guarantee, a function it exited some time back, in the changed environment. (ii) Since 2014, there has been some effort to professionalise DICGC. This need to be taken forward. (iii) The anomalous situation arising from commercial banks meeting the cost of the inefficiency of cooperative banks will have to be rectified. (iv) The corporation should apply its mind as to whether continuing the level of deposit insurance should be retained at a low of Rs one lakh. Perhaps the threshold should be raised to a level to provide cover for at least 50 per cent of bank deposits. (v) Corporation could consider expanding its ambit to all financial institutions regulated by RBI which are accepting deposits from public. This may need differential rates of premia and coordination with regulatory and supervisory arms of RBI. M G Warrier,Mumbai February 17, 2016 Stop speculating on RBI Apropos ?RBI Guv and Never Ending Speculation? (Economic Times, Money & banking, February 17), one wonders why ET should join the speculators. It is common knowledge that even in 2013 when Dr Raghuram Rajan was being tipped for the Governor?s position, there were other influential contenders and an opposing school of economists who were spreading all sorts of rumours against Dr Rajan. They were again active during the first half of 2014, gossiping about the possibility of a change of guard at Mint Road, post General Election. It is India?s good fortune that Dr Rajan survived the moves against him, so far. World has acknowledged that the leadership provided by Dr Rajan in the conduct of RBI?s affairs has been excellent. If India decides and Dr Rajan accepts on a longer term re-appointment, this paper should not have a doubt about Dr Rajan?s capability to sort out his personal relationship with the University with which he has had an enduring relationship for the last two decades. As any new incumbent governor in RBI takes minimum six months to ?settle down? and the ?work-in-progress? there now needs continuity in leadership, GOI may not think in terms of replacing Dr Rajan in September 2016 and perhaps offer a re-appointment for a longer tenure. But, reports like this can help gossip-mongers and have some impact on the thought processes of economists and media which can influence economy adversely. M G WARRIER, Mumbai Business Standard February 26, 2016 Banks? health worries This refers to Sudhir Keshav Bhave?s letter ?A sinking ship? (February 26). This could be an example of how transparency in a sensitive business like banking can spread panic. The Indian banking sector is not in as bad a shape as is being made out by some analysts and external agencies. Major Indian commercial banks including SBI have been able to meet all statutory requirements. Unlike their corporate co-travellers, banks are meeting their payment obligations on due dates and in the recent past there have been no bank failures in the commercial banking sector in India. Part of credit for this should go to the vigilant regulator. This is not to argue that all is well as regards functioning of commercial banks. There is immediate need to restore the health of the banking system impaired mainly by reluctance of big borrowers to make timely repayment and heavy burden on public sector banks (PSBs) arising from workload and drain on resources in performance of social responsibilities. There is no point in arguing now that the overhaul and professionalization of public sector banks (PSBs) should have happened along with bank nationalisation and there should have been regular ?health checks? and ongoing corrections. Just as a ?health check-up? does not change the condition of a person, the re-classification of more loans as NPAs does not alter a bank?s ability to change. The need of the hour is to support banks to recover their dues from borrowers who have the capacity to repay, infuse professionalism in the banks? working and restore the faith in the banking system. As private sector banks have failed to perform their responsibilities and are not too willing to grow (their share in banking business is less than 30 per cent), privatising the existing public sector banks is no solution. Perhaps, GOI should consider nationalising entire banking business and restructuring the banking system to serve public interest. M G Warrier, Mumbai III SELF DEVELOPMENT A suggestion shared by a doctor, worth accepting* Each individual must take note of the three ? minutes. Why is it important? Three ? minutes will greatly reduce the number of sudden deaths. Often this occurs when a person who still look healthy, has died in the night. We hear stories of people suddenly die? The reason is that when you wake up at night to go to the bathroom, it is often done in a rush. Immediately when stand up, the brain lacks blood flow. Why "three ? minutes" is very important? In the middle of the night when you are awakened by the urge to urinate for example, ECG pattern can change. Because getting up suddenly, the brain which is anaemic can lead to heart failure due to lack of blood. Hence, always practice " three ? minutes", which are: 1. When waking from sleep, lie in bed for the first ? minute; 2. Sit in bed for the next ? minute; 3. Lower your legs, sitting on the edge of the bed for the last ? minute. After three ? minutes, you will not have an anaemic brain and heart will not fail, reducing the possibility of a fall or sudden death. Share with family, friends loved ones. It can occur regardless of age; young or old. Sharing is Caring. If you already know, regard this as refresher! *Copied from an email received from Vathsala Jayaraman (Exrbites Group) __._,_.___ 999990323 27-Feb-2016 http://mgwarrier.blogspot.com/2016/02/weekend-lighter-feel-hunger.html')


a='U as df srg  fsa chandru@gmail.com sdf sdf sd f'
a.find('chandru')
#NER

train2= [('the chandru@gmail.com sdfsf  sdf sdf',{'entities': [(4, 21,'emailid')]}), 
        ("Ssdf chandru@gmail.com sdf fhg gd dfg df  dfg ",{'entities': [(5, 22,'emailid')]}), 
        ('Contemporary chandru@gmail.com sdf ds dfg fd g sdfs',{'entities': [(13, 30,'emailid')]}),
        ('A mixedmedia design pairs df chandru@gmail.com sdf sd fdg sdf sd',{'entities': [(29, 46,'emailid')]}),
        ("U as df srg  fsa chandru@gmail.com sdf sdf sd f", {'entities': [(17, 34,'emailid')]})]

import random



def train_spacy(data,iterations):
    train2 = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in train2:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(train2)
            losses = {}
            for text, annotations in train2:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(train2, 20)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
#h=''
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text,ent.label_)
    
    
    
######
import scispacy
import spacy
#import en_core_sci_sm   #The model we are going to use
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
#nlp = en_core_sci_sm.load() 
import en_core_sci_lg

nlp = en_core_sci_lg.load()


text = '''Myeloid derived suppressor cells  are immature 
          myeloid cells with immunosuppressive activity. 
          They accumulate in tumor-bearing mice and humans 
          with different types of cancer, including hepatocellular 
          carcinoma .'''
doc = nlp(text)
print(doc.ents)

#sentences2 = [sent_tokenize(document) for document in train['text']]
'''
nlp.max_length = 1268222 # or even higher

train1.info()
train_review = train['text'].iloc[:10]
train_token = ''
for i in train['text']:
   train_token += str(i)
   
doc = nlp(train_token)
print(doc)
print(list(doc.sents))
print(doc.ents)

abbreviation_pipe = AbbreviationDetector(nlp)
print(doc._.abbreviations)

nlp.add_pipe(abbreviation_pipe)
#Print the Abbreviation and it's definition

doc = nlp(text)
print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
      print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")'''
      
      
      
#####SENTIMENT ANALYSIS

#train3=train['text']
from nltk.corpus import opinion_lexicon

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn

##Unsupervised
#using textblob
import textblob

def score(text):
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity
def predict(text):
    train['score']=train['text'].apply(score)
    return(train)
    
train3=predict(train)  
train3.info()  
train3=train3.drop(['Patient Or not'],axis=1)
train3['Sentiment']=['positive' if score >=0 else 'negative' for score in train3['score']]


## Evaluation of performance#cannot be done
afn = Afinn(emoticons = True)
train.info()
#train=train.drop(['score'],axis=1)
afn.score("I love it")
x_afinn=pd.DataFrame(train)
def score(text):
    from afinn import Afinn
    return afn.score(text)
def predict(text):
    x_afinn['score']=x_afinn['text'].apply(score)
    return(x_afinn)
x1_afinn=predict(x_afinn)
x1_afinn['Sentiment']=['positive' if score >=0 else 'negative' for score in x1_afinn['score']]
x1_afinn.info()
x1_afinn=x1_afinn.drop(['Patient Or not'],axis=1)
#sentiment analyzing using vader model
#train=train.drop(['score','Sentiment'],axis=1)
x_vader=pd.DataFrame(train)
def score(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader=SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound']
def predict(text):
    x_vader['score']=x_vader['text'].apply(score)
    return(x_vader)
x1_vader=predict(x_vader)
x1_vader['Sentiment']=['positive' if scores>=0 else 'negative' for scores in x1_vader['score']]

x1_vader=x1_vader.drop(['Patient Or not'],axis=1)


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
x1_vader['score'].plot(
    kind='hist',
    bins=50,
    )

###########

######WORD FREQUENCY
import itertools
import collections
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in train['text']]


all_words = list(itertools.chain(*tokenized_corpus))

counts_no = collections.Counter(all_words)

counts_no.most_common(15)

cleanwords= pd.DataFrame(counts_no.most_common(15),
                             columns=['words', 'count'])

cleanwords.head()



fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
cleanwords.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words  (Including All Words)")

plt.show()


nltk.download('averaged_perceptron_tagger')
pos = nltk.pos_tag(all_words)
print(pos)
    ###SENTENCE FREQUENCY
#sentence frequency
import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist


train_token1 = ''
for i in train['text']:
   train_token1 += str(i)
   
 

data_analysis = nltk.FreqDist(train['text'])
 
# Let's take the specific words only if their frequency is greater than 3.
filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 3])
 
for key in sorted(filter_words):
    print("%s: %s" % (key, filter_words[key]))
 
data_analysis = nltk.FreqDist(filter_words)
 
data_analysis.plot(25, cumulative=False)




#######BOW,TFIDF,WORD2VEC


#BOW
x1=train['text']
x1.dropna(inplace=True)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary = False, min_df = 0.01,max_df = 0.50, ngram_range=(1,2))
cv_train_features = cv.fit_transform(x1)


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,25):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)#init-initialization=kmeans++ to avoid random initialization trap that is choosing the wrong centroids
    kmeans.fit(cv_train_features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,25),wcss)

kmeans =KMeans(n_clusters=4,init='k-means++',random_state=1234)
y_kmeans=kmeans.fit_predict(cv_train_features)
y_kmeans=pd.DataFrame(y_kmeans)
train_bow=pd.concat([x1,y_kmeans],axis=1)

#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df = 0.01,max_df = 0.50,ngram_range=(1,2))#unigrams and bigrams
tv_train_features = tv.fit_transform(x1)


kmeans =KMeans(n_clusters=4,init='k-means++',random_state=1234)
y_kmeans_tfidf=kmeans.fit_predict(tv_train_features)
y_kmeans_tfidf=pd.DataFrame(y_kmeans_tfidf)
train_tfidf=pd.concat([x1,y_kmeans],axis=1)

tffrequency=dict(zip(tv.get_feature_names(), tv.idf_))
print(tffrequency)


#PCA

'''Before attempting to cluster the data, we will usually want to reduce the dimensionality of the data because this helps to mitigate the problem of overfitting. Note the distinction between the two terms:

Dimensionality reduction: find the linear combinations of variables that are most 'interesting' in the data. For example, the popular PCA technique finds linear transformations of input features that maximize the variance of the data points along the new axes.

Clustering: find data points that can be grouped together as separate classes.'''




#features(columns) is in order of descending sort


from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2,kernel='rbf')
x_pca=kpca.fit_transform(tv_train_features)

n_clusters=3
km_model = KMeans(n_clusters=n_clusters, max_iter=10, n_init=2, random_state=0)

# K-means (from number of features in input matrix to n_clusters)
km_model.fit(x_pca)
df_centers = pd.DataFrame(km_model.cluster_centers_, columns=['x', 'y'])

plt.figure(figsize=(4,4))
#plt.suptitle('PCA features colored by class; grey circles show the k-means centers')
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=km_model.labels_, s=50, cmap='jet')
#plt.scatter(df_centers['x'], df_centers['y'], c='grey', s=500, alpha=0.2);
'''
dy = 0.04
for i, txt in enumerate(km_model.labels_):
    plt.annotate(txt, (x_pca[i, 0], x_pca[i, 1] + dy))'''

####


#SENTIMENT POLARITIES VISUALIZATION
#for vader
x1_vader['score'].plot(
    kind='hist',
    bins=50,
    )

#for afinn
x1_afinn['score'].plot(
    kind='hist',
    bins=50,
    )

#for textblob
train3['score'].plot(
    kind='hist',
    bins=50,
    )


#WORD2VEC
from gensim.models import Word2Vec

size = 1000
window = 3
min_count = 1
workers = 3#core parellization
sg = 0#1-sg ,0-CBOW


w2v_model = Word2Vec(tokenized_corpus, min_count = min_count, size = 30, workers = workers, window = window, sg = sg)


max_dataset_size_new = len(w2v_model.wv.syn0)
words = w2v_model.wv.vocab
x_w2v=w2v_model[w2v_model.wv.vocab]
print(x_w2v)

##WORD CLOUD

    
Topwords=w2v_model.wv.index2entity[:20]
print(Topwords)
Topwords =set(Topwords)


bottomwords=w2v_model.wv.index2entity[-20:]
print(bottomwords)
bottomwords =set(bottomwords)




import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image
dataset =set(all_words[:100])

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


##WORDCLOUD FOR THE OVERALL DATASET
show_wordcloud(dataset)

#WORD CLOUD TOPWORDS
show_wordcloud(Topwords)

##WORD CLOUD BOTTOMWORDS

show_wordcloud(bottomwords)


####TSNE
from sklearn.manifold import TSNE
tokenized_corpus1=tokenized_corpus[:100]
w2v_model1 = Word2Vec(tokenized_corpus1, min_count = min_count, size = 30, workers = workers, window = window, sg = sg)

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in w2v_model1.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(w2v_model1)




#####Summarization TECHNIQUES
#ABSTRACTIVE Summarization

train_token =  """
                I'm just saying that Entresto has been a major failure of Novartis commercial organisation.
                Factors such as rising obese population, growing incidence of heart diseases, chronic conditions, and diabetes, and increasing hypertensive patients have accelerated demand for blood pressure monitoring devices globally.
                Let me assure you that I'd rather have a coke, gummi bears, and a bag of cheez doodles than a pack of cigs right now.
                Blood pressure is a measurement of the pressure in your arteries during the active and resting phases of each heartbeat.
                sometimes from one heartbeat to the next, depending on body position, breathing rhythm, stress level, physical condition, medications you take, what you eat and drink, and even time of day.
                Most doctors consider chronically low blood pressure too low only if it causes noticeable symptoms.
                Losing a lot of blood from a major injury or internal bleeding reduces the amount of blood in your body, leading to a severe drop in blood pressure.
                I try to move as much as I can everyday to feel useful, trying to pull my weight.
                Having congestive heart failure really has taken it toll on me physically.
                We tend to think of both mental health and drug addiction to be issues at the margins of society.
                Heart failure is an important and ever expanding sub-speciality of cardiology.
                This updated book comprehensively covers all aspects necessary to manage a patient with heart failure."""



import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

preprocess_text = train_token.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


##EXTRACTIVE SUMMARIZATION


'''train4=train
train4=train4[:100]
train_token1 = ''
for i in train_token1:
   train_token1 += str(i)

print(train_token1)
sentence_list = nltk.sent_tokenize(train_token)
print(sentence_list)'''


train_token =  """
                I'm just saying that Entresto has been a major failure of Novartis commercial organisation.
                Factors such as rising obese population, growing incidence of heart diseases, chronic conditions, and diabetes, and increasing hypertensive patients have accelerated demand for blood pressure monitoring devices globally.
                Let me assure you that I'd rather have a coke, gummi bears, and a bag of cheez doodles than a pack of cigs right now.
                Blood pressure is a measurement of the pressure in your arteries during the active and resting phases of each heartbeat.
                sometimes from one heartbeat to the next, depending on body position, breathing rhythm, stress level, physical condition, medications you take, what you eat and drink, and even time of day.
                Most doctors consider chronically low blood pressure too low only if it causes noticeable symptoms.
                Losing a lot of blood from a major injury or internal bleeding reduces the amount of blood in your body, leading to a severe drop in blood pressure.
                I try to move as much as I can everyday to feel useful, trying to pull my weight.
                Having congestive heart failure really has taken it toll on me physically.
                We tend to think of both mental health and drug addiction to be issues at the margins of society.
                Heart failure is an important and ever expanding sub-speciality of cardiology.
                This updated book comprehensively covers all aspects necessary to manage a patient with heart failure."""

sentence_list = nltk.sent_tokenize(train_token)

stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}
for word in nltk.word_tokenize(text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1


maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    

    
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 50000:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)

#ngrams
import nltk
from nltk.util import ngrams
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

    print("1-gram: ", extract_ngrams(train_token, 1))
print("2-gram: ", extract_ngrams(train_token, 2))
print("3-gram: ", extract_ngrams(train_token, 3))
print("4-gram: ", extract_ngrams(train_token, 4))


####MODELL
train.info()
train=train.drop(['score','Sentiment'],axis=1)

patient=train['Patient Or not']
train=train.drop(['Patient Or not'],axis=1)
#preprocess test
stop_words = nltk.corpus.stopwords.words('english')

test['text']= test['Conversations'].map(lambda s:preprocess(s)) 
test.info()
test=test.drop(['Conversations'],axis=1)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,patient,test_size=0.2,random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
'''max_df = 25 means ignore terms that appear in more than 25 documents".
min_df = 5 means "ignore terms that appear in less than 5 documents".'''
tv1 = TfidfVectorizer(ngram_range=(1,2))#unigrams and bigrams
tv_train_features1 = tv1.fit_transform(x_train['text'])
tv_train_features2 = tv1.fit_transform(test['text'])

#print(tv_train_features1)
tv_test_features1 = tv1.transform(x_test['text'])
#print(tv_test_features1)

##logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l2',max_iter = 500,C= 1,solver = 'lbfgs')
lr.fit(tv_train_features1,y_train)
lr_predictions= lr.predict(tv_test_features1)
lr_predictions

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
         
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
accuracy_score(y_test,lr_predictions)



#Naive baye's
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
tv_train_features1=tv_train_features1.toarray()

classifier.fit(tv_train_features1,y_train)
gaussian_pred=classifier.predict(tv_test_features1)

from sklearn.metrics import confusion_matrix, classification_report

gau=confusion_matrix(y_test,gaussian_pred)
print(gau)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,gaussian_pred)

print(classification_report(y_test,gaussian_pred))



#random forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier_rf.fit(tv_train_features1,y_train)

rf_pred=classifier.predict(tv_test_features1)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

confusion_matrix(y_test,rf_pred)
rf=accuracy_score(y_test,rf_pred)
print(rf)

print(classification_report(y_test,rf_pred))


#SVM

from sklearn.svm import SVC
classifier_sv=SVC(kernel='rbf',random_state=0)
classifier_sv.fit(tv_train_features1,y_train)
print(classifier_sv)
sv_pred=classifier.predict(tv_test_features1)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

svm=confusion_matrix(y_test,gaussian_pred)
print(svm)

accuracy_score(y_test,sv_pred)

print(classification_report(y_test,sv_pred))

#K fold cross validation
from sklearn.model_selection import cross_val_score
#cross=tv1.fit_transform(x_train['text'])
accuracies=cross_val_score(estimator=classifier_sv,X=tv_train_features1,y=y_train,cv=10)

print("Accuracy:{:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation:{:.2f} %".format(accuracies.std()*100))


#HYPER PARAMETER TUNING

from sklearn.model_selection import GridSearchCV 

param_grid = {'C': [10, 100, 1000],  
              'gamma': [1,0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(tv_train_features1, y_train) 

print(grid.best_params_) 
#C=100,GAM=0.01,ker=rbf best

grid_predictions = grid.predict(tv_test_features1) 

print(classification_report(y_test, grid_predictions)) 
accuracy_score(y_test,grid_predictions)#92 percent



###ML PIPELINE

from sklearn.pipeline import make_pipeline
import os
import pickle



    
filename='nlp_model.pkl'
pickle.dump(grid.best_estimator_,open(filename,'wb'))


#with open(Pkl_Filename, 'rb') as file:  
 #   Pickled_LR_Model = pickle.load(file)

#Pickled_LR_Model.fit(tv_train_features1, y_train) 
#grid_predictions 1= Pickled_LR_Model.predict(tv_test_features1) 

###PREDICTION For TEST DATASET GIVEN
from sklearn.ensemble import RandomForestClassifier
classifier_rf=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

tv_train_features3=tv_train_features1
tv_train_features3=pd.DataFrame(tv_train_features3)
tv_train_features3=tv_train_features3.iloc[:,:29357]
classifier_rf.fit(tv_train_features3,y_train)

rf_pred1=classifier.predict(tv_test_features1)

rf_pred1=pd.DataFrame(rf_pred1)

rf_pred1.to_csv('grid_predictions_test.csv')


    
#HYPER PARAMETER TUNING FOR RANDOM FOREST
from sklearn.model_selection import GridSearchCV
parameter=[{'n_estimators':[50,100,500,1000],'criterion':['gini','entropy'],'bootstrap':[True],'max_depth':[2,3,4,5,6]}]

grid_search=GridSearchCV(classifier_rf,param_grid=parameter,scoring='accuracy',cv=10)

grid_search=grid_search.fit(tv_train_features1, y_train)


##PREDICTION COMPARISON
metrics=pd.DataFrame(index=['accuracy','precision','recall','F1 score'],
                     columns=['logistic','Naive Bayes','Random forest','SVM(grid)'])

metrics.loc['accuracy','logistic']=accuracy_score(y_test,lr_predictions)
metrics.loc['precision','logistic']=precision_score(y_test,lr_predictions)
metrics.loc['recall','logistic']=recall_score(y_test,lr_predictions)
metrics.loc['F1 score','logistic']=f1_score(y_test,lr_predictions)

metrics.loc['accuracy','Naive Bayes']=accuracy_score(y_test,gaussian_pred)
metrics.loc['precision','Naive Bayes']=precision_score(y_test,gaussian_pred)
metrics.loc['recall','Naive Bayes']=recall_score(y_test,gaussian_pred)
metrics.loc['F1 score','Naive Bayes']=f1_score(y_test,gaussian_pred)

metrics.loc['accuracy','Random forest']=accuracy_score(y_test,rf_pred)
metrics.loc['precision','Random forest']=precision_score(y_test,rf_pred)
metrics.loc['recall','Random forest']=recall_score(y_test,rf_pred)
metrics.loc['F1 score','Random forest']=f1_score(y_test,rf_pred)


metrics.loc['accuracy','SVM(grid)']=accuracy_score(y_test, grid_predictions)
metrics.loc['precision','SVM(grid)']=precision_score(y_test, grid_predictions)
metrics.loc['recall','SVM(grid)']=recall_score(y_test, grid_predictions)
metrics.loc['F1 score','SVM(grid)']=f1_score(y_test, grid_predictions)

100*metrics
fig,ax=plt.subplots(figsize=(8,5))
metrics.plot(kind='bar',ax=ax)
ax.grid();



#deployment
import pickle
from sklearn.externals import joblib
cv=CountVectorizer()
X=cv.fit_transform(x1)
x3=pd.DataFrame(X)
pickle.dump(cv,open('transform.pkl','wb'))

X.shape()
from sklearn.pipeline import make_pipeline
import os
import pickle

tv_train_features4=tv_train_features1
tv_train_features4=pd.DataFrame(tv_train_features1)
tv_train_features4=tv_train_features4.iloc[:,:19930]

classifier_rf1=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier_rf1.fit(tv_train_features4,y_train)

rf_pred1=classifier.predict(tv_test_features1)
filename='nlp_model.pkl'
pickle.dump(classifier_rf1,open(filename,'wb'))
