import nltk
from nltk.collocations import *
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import string
import itertools

class Lar:
    filter = {}
    def retrieveText(self, dataFileDict, textCol='DocText', yearCol='Y', monthCol='M', dateCol='D', 
    removePunctuation=True, lowercase=True, removeDigits=True, calculateDate=False, pickleOverride=''): 
        
        
        dataFiles = sorted(dataFileDict, key=dataFileDict.__getitem__)
        
        ## Use pickle override
        if pickleOverride != '':
            docs = read_pickle( pickleOverride)
            print('Loaded override custom pickle - this should have columns for keywords and securitizing/desecuritizing counts/pulses.' )
        else: 

            docs={}
            for dataSrc in dataFiles:
                fileSrc = dataFileDict[dataSrc]
                print('--Loading file from scratch:', dataSrc, '---', fileSrc)
                dataChunks = []
                for chunk in pd.read_csv(fileSrc, sep=',', chunksize=5000):
                    dataChunks.append(chunk)

                docs[dataSrc] = pd.concat(dataChunks, axis= 0)
                docs[dataSrc]['DataSrc'] = dataSrc
                
                
                del dataChunks

            #combine docs
            docs = pd.concat( docs[dataSrc] for dataSrc in dataFiles)
            docs = docs.rename(columns={textCol: 'DocText', yearCol: 'Y', monthCol: 'M', dateCol: 'D'})
            
            print('--Combined length: ',len(docs))
            #PROCESS
            print('--Starting cleanText method')
            docs = self.cleanText(docs,removePunctuation=removePunctuation, 
                                        lowercase=lowercase, removeDigits=removeDigits, calculateDate=calculateDate)
            
            docs.reset_index(inplace = True)
            docs.drop(['index'], axis=1,inplace = True)

            #create a filter
            for dataSrc in dataFiles:
                self.filter[dataSrc] = docs['DataSrc']==dataSrc
            
            
        return docs

    def addKeywordColumn(self, docs, keywordList, exactMatch=False):
        
        for keyword in keywordList:
            
            if exactMatch: 
                print('--EXACT MATCH keyword:', keyword)
                if len(keyword.split())>1:
                    docs['kexact_' + '_'.join(keyword.split())]= docs.apply(lambda x: x['DocText'].lower().count(keyword) ,axis=1)
                else:
                    docs['kexact_' + keyword]= docs.apply(lambda x: x['DocText'].lower().split().count(keyword) ,axis=1)
                self.filter[keyword] = docs['kexact_'+'_'.join(keyword.split())]>0
            else :
                print('--Word begins with: ', keyword, '(this allows variations like', keyword+'s and',keyword+'ed)' )
                if len(keyword.split())>1:
                    docs['kstarts_' + '_'.join(keyword.split())]= docs.apply(lambda x: x['DocText'].lower().count(keyword) ,axis=1)
                else: 
                    docs['kstarts_' + keyword]= docs.apply(lambda x: sum(item.startswith(keyword) for item in x['DocText'].lower().split()) ,axis=1)
                self.filter[keyword] = docs['kstarts_'+'_'.join(keyword.split())]>0
        
        return docs


    def cleanText(self, docs,  removePunctuation=False, lowercase=False, removeDigits=False, calculateDate=False):
        
        print('--Drop duplicates')
        docs = docs.drop_duplicates()
        
        
        docs.Y = docs.Y.astype(int)
        docs.M = docs.M.astype(int)
        docs.D = docs.D.astype(int)
        
        

        if calculateDate: 
            #add date column
            print('--Add date column')
            docs['FullDate']= docs.apply(lambda x:datetime.strptime("{0} {1} {2}".format(x['Y'],x['M'], x['D']), "%Y %m %d"),axis=1)
            docs = docs.drop(['Y','M','D'], axis=1)

            #reorder, move fulldate to the beginning
            DateCol = docs["FullDate"]
            docs.drop('FullDate', axis=1, inplace=True)
            docs.drop('Unnamed: 0', axis=1, inplace=True)
            docs.insert(1,'FullDate',DateCol)

        if removePunctuation:
            print('--Remove punctuation')
            translator = str.maketrans({key: None for key in '!"#$%&\'()*+,.;<>?@[]^`{|}~'})
            docs['DocText']= docs.apply(lambda x:  x['DocText'].translate(translator) ,axis=1)
            translator = str.maketrans({key: ' ' for key in '/:\\-=_'})
            docs['DocText']= docs.apply(lambda x:  x['DocText'].translate(translator) ,axis=1)
        
        if removeDigits:
            print('--Remove digits')
            translator = str.maketrans({key: None for key in string.digits})
            docs['DocText']= docs.apply(lambda x:  x['DocText'].translate(translator) ,axis=1)
        
        if lowercase:
            print('--Lowercase')
            docs['DocText']= docs.apply(lambda x:  x['DocText'].lower() ,axis=1)
        
        print('--Number of words')
        docs["nwords"] = docs.apply(lambda x: len(x['DocText'].split()) ,axis=1)
        
        return docs

    
    def collocate(self, docs, label='', ngram=2, keyword='', freqFilter=3, windowSize=3, removeStop=True, save=False):
        
        wordBagFile = 'wordBagFile--label'+ label +'-removeStop'+ ('1' if removeStop==True else '0') + '.csv'
        pd.options.mode.chained_assignment = None
        print('--Attempting file:', wordBagFile)
        
        try:
            wordBagFrame=read_csv('wordBags/'+wordBagFile, encoding="utf-8")
            wordBag = wordBagFrame.iloc[0]['col'].split()
            print('--File for', label, 'loaded into memory. Found', len(wordBag), 'words')
        except: 
            print('--' + label, 'Wordbag file not found-starting from scratch!')
            customStopWords = []
            if removeStop:
                text_file = open("RSmartStopWords.txt", "r", encoding="utf")
                customStopWords = text_file.read().splitlines()
                #need to remove punctuation from stopwords, because punctuation has already been removed from main text
                customStopWords = [''.join(c for c in s if c not in string.punctuation) for s in customStopWords]
                print('--Stop words loaded. ')
            
            docs['DocTextWordbag'] = docs['DocText'].apply(lambda x: [item for item in x.split() if item not in customStopWords])
            print('--Stop words removed')
            wordBag = list(itertools.chain(*docs.DocTextWordbag))
            #wordBag = docs.DocTextsStopless.sum()
            print('--Word bag created')
            if save: 
                wordsJoined = ' '.join(wordBag)
                oneCellDf = pd.DataFrame({'col':[wordsJoined]})
                oneCellDf.to_csv('wordBags/'+wordBagFile, encoding="utf-8")
                #np.savetxt('wordBags/test.txt', wordsJoined, delimiter=" ", fmt="%s", encoding="utf-8")
                print('--Word bag saved to:', wordBagFile)
        
        print('--Building finder for ngram =', ngram)
        if ngram==2:
            finder = BigramCollocationFinder.from_words(wordBag, window_size = windowSize)
        elif ngram==3: 
            finder = TrigramCollocationFinder.from_words(wordBag, window_size = windowSize)
        print('--Finder created')
        
        # only bigrams that appear 3+ times
        finder.apply_freq_filter(freqFilter)
        print('--Finder frequency filter applied')
        
        # only bigrams that contain keyword
        collocateKeyword = lambda *w: keyword not in w
        finder.apply_ngram_filter(collocateKeyword)
        print('--Finder keyword filter applied')
        # return the 10 n-grams with the highest PMI
        #print(finder.nbest(bigram_measures.likelihood_ratio, 20))
        print('\n')
        return finder
    
    def collatedTable(self, collations, scoreType='LL'):
        
        ngrams = collations.default_ws
        if ngrams==2:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            collationsList = collations.score_ngrams(bigram_measures.likelihood_ratio) if scoreType=='LL' else collations.score_ngrams(bigram_measures.pmi) 
        else:
            trigram_measures = nltk.collocations.TrigramAssocMeasures()
            collationsList = collations.score_ngrams(trigram_measures.likelihood_ratio) if scoreType=='LL' else collations.score_ngrams(trigram_measures.pmi) 
        
        if len(collationsList) > 0:
            tempDF =  pd.DataFrame(collationsList)
            wordsDF = pd.DataFrame(list(tempDF.ix[:,0])) 
            scoreDF = pd.DataFrame({scoreType:tempDF.iloc[:,-1]})
        return pd.concat([wordsDF, scoreDF], axis=1)
        #return scoreDF


    def count(self, DocTextCol, ngramRange=(1,3), returnRange=100, removeStop=True):
        customStopWords = []

        if removeStop:
            text_file = open("RSmartStopWords.txt", "r", encoding="utf")
            customStopWords = text_file.read().splitlines()

        vectorizer = CountVectorizer(min_df=1, stop_words=customStopWords,analyzer="word",ngram_range=ngramRange)
        count_matrix = vectorizer.fit_transform(DocTextCol)

        cm_feature_names = vectorizer.get_feature_names()
        #len(cm_feature_names)
        #cm_feature_names[0:30]
        termFreq = list(zip(cm_feature_names, np.asarray(count_matrix.sum(axis=0)).ravel()))
        cm_sorted_phrase_scores = sorted(termFreq, key=lambda t: t[1] * -1)[:returnRange]
        return cm_sorted_phrase_scores


    def compare(self, df1, df2, lab1='DF1', lab2='DF2', retainKeyword='', print=False, unique=False):
        df1 = df1.copy()
        df2 = df2.copy()
        df1 = df1[df1.iloc[:, -2].str.contains(retainKeyword)].reset_index(drop=True)
        df2 = df2[df2.iloc[:, -2].str.contains(retainKeyword)].reset_index(drop=True)
        
        ngrams = 2 if len(df1.columns) ==3 else 3

        if unique:
            
            deleteIndexDf1 = []
            deleteIndexDf2 = []
            for i, row in df1.iterrows():
                if ngrams==2:
                    deleteIndexes = (df2.index[(df2[0] == row[0]) & (df2[1] == row[1])].tolist())
                elif ngrams==3:
                    deleteIndexes = (df2.index[(df2[0] == row[0]) & (df2[1] == row[1]) & (df2[2] == row[2])].tolist())

                if len(deleteIndexes) > 0:
                    if ngrams==2:
                        deleteIndexDf1.extend((df1.index[(df1[0] == row[0]) & (df1[1] == row[1])].tolist()))
                    elif ngrams==3:
                        deleteIndexDf1.extend((df1.index[(df1[0] == row[0]) & (df1[1] == row[1]) & (df1[2] == row[2])].tolist()))
                    
                    deleteIndexDf2.extend(deleteIndexes)

            df1.drop(df1.index[deleteIndexDf1], inplace=True)
            df2.drop(df2.index[deleteIndexDf2], inplace=True)
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            
        df1.columns = [lab1 + '_' + str(col) for col in df1.columns]
        df2.columns = [lab2 + '_' + str(col) for col in df2.columns]
        
        compareFrame = pd.concat([df1, df2 ], axis=1) 
        compareFrame.fillna('', inplace=True)

        
        if print: 
            fileName = 'compareCollate-' + lab1 + '-' + lab2 + '-unique'+ ('1' if unique==True else '0')+'.xlsx'
            compareFrame.to_excel(fileName)
            os.startfile(fileName, "print")

        return compareFrame        
        


    def frameCount(self, docs, keywordList):
        print('--Frame count')
        return docs.map(lambda x:  sum(any(keyword in DocTextWord for keyword in keywordList) for DocTextWord in x.lower().split()))
        
    def framePulse(self, frameCol  ):
        print('--Frame pulse')
        return frameCol.map(lambda x: 1 if x > 0 else 0)


    #this returns a periodized dataframe of how frequently words occur overall and per 1000 words 
    def periodize(self, df, keywords, period, wordsPer, prefix=''):

        #
        #periodsOnlyDf = (df.groupby(df.FullDate.dt.to_period(period)).sum()).to_frame()
        #total words for the period 
        periodizedDf = (df['nwords'].groupby(df.FullDate.dt.to_period(period)).sum()).to_frame()
        
        if len(keywords)>0:
            for keyword in keywords:
                print('--Starting keyword:', keyword)
                keywordCol = keyword if keyword.startswith('kexact_')  else 'kstarts_' + keyword
                
                periodizedDf[keyword] = (df[keywordCol].groupby(df.FullDate.dt.to_period(period)).sum()).to_frame()

                # calculate WPT
                periodizedDf[keyword+'_wpt'] = (periodizedDf[keyword] * wordsPer) / periodizedDf['nwords']
        else: 
            for col in df.columns:
                if col.startswith('kexact_') or col.startswith('kstarts_') or col.endswith('_count') or col.endswith('_pulse'):
                    print('--Starting column:', col)
                    
                    periodizedDf[col] = (df[col].groupby(df.FullDate.dt.to_period(period)).sum()).to_frame()

                    # calculate WPT
                    periodizedDf[col+'_wpt'] = (periodizedDf[col] * wordsPer) / periodizedDf['nwords']

        if prefix != '':
            periodizedDf.columns = [prefix + str(col) for col in periodizedDf.columns]
        return periodizedDf
