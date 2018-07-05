## Linguistics Analysis for Research

Description
===========

Analyze corpora of texts (e.g., news media articles, speeches) to see trends in keyword usage, salience and context over time. 

Features include: 
- Import content from multiple CSV files into a single dataframe object
- Text preprocessing (remove punctuation, lowercase, remove numbers, calculate date from M, D, Y fields)
- Count keywords in each document
- Determine presence and extent of frame (looks for a match in a list of keywords)
- Aggregate keywords/frames into time-series (e.g., annual or quarterly count of keyword)
- Explore collocations (bigram and trigrams) of keywords to understand their context, as well as collocation scores (log likelihood or PMI) 
- Count most common words in entire corpus


Example Usage
===========
First import the required items. Because the package uses dataframes, you should import pandas. 
```python
from nqlar import Lar
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt 
```

Instantiate the Lar class and create a variable for the filter dictionary (f)
```python
lar = Lar()
f = lar.filter 
```

Create a dictionary that describes the source data. The key should be a short label (e.g., CNN, FOX, NYT) and the value should point to the file. Currently, the data must be stored in CSV files. 
```python
dataFileDict = {
    'CNN':'CNN-Economy.csv',
    'FOX':'Fox-Economy.csv'
}
```

Pass the dictionary to the retrieveText function, along with some other options and data maps:
- textCol: the name of the variable in the CSV file that contains the text to be analyzed
- yearCol (optional): the name of the variable in the CSV file that contains the year of the document
- monthCol (optional): the name of the variable in the CSV file that contains the month of the document
- dateCol (optional): the name of the variable in the CSV file that contains the date of the document
- removePunctuation: boolean to remove punctuation
- lowercase: boolean to lowercase all text
- removeDigits: boolean to remove all digits
- calculateDate: boolean to calculate date

The retrieveText function returns a dataframe of the text and metadata. 

```python
docsDf = lar.retrieveText(dataFileDict,textCol='Text', yearCol='Y', monthCol='M', dateCol='D', 
                              removePunctuation=True, lowercase=True, removeDigits=True, calculateDate=True)
```


The addKeywordColumn function takes the document dataframe and a list of keywords. It returns a dataframe with new columns for each keyword that identifies the number of times that keyword occurs for each record (or document). Optional parameter exactMatch=True will search exact matches, while exactMatch=False will only search for words that start with the keyword passed. 
```python
keywordCols = ['economy','money','finance']
docsDf = lar.addKeywordColumn(docsDf,keywordCols, exactMatch=False )
```

Use plotting functionality to see how frequently a keyword occurs by year
```python
docsDf[f['economy']].groupby(docsDf['FullDate'].map(lambda x: x.year)).size().plot(kind="bar", figsize=(20,10), 
    title="Keyword Frequency");
```

Create time-series of keyword counts, both in absolute as well as word-per-thousand.  
```python
annualPeriodized = {}
keywordsForPeriodize = []
annualPeriodized['ALL'] = lar.periodize(df=docsDf, keywords=keywordsForPeriodize, period='a', wordsPer=1000)
for src in dataFileDict.keys():
    print('\nSource:', src)
    annualPeriodized[src] = lar.periodize(df=docsDf[f[src]], keywords=keywordsForPeriodize, period='q', wordsPer=1000)
    
    # Relabel the columns to not have the "kstarts_" prefix
    annualPeriodized[src].columns = [str(col).split("kstarts_")[1] if 'kstarts_' in str(col) else str(col) for col in annualPeriodized[src].columns]
```

Use plots to chart time-series 
```python
annualPeriodized['CNN']['economy'].plot(figsize=(15,10), linewidth="3", color='red', label='CNN');
annualPeriodized['FOX']['economy'].plot(figsize=(15,10), linewidth="3", color='green', label='FOX');
plt.legend(prop={'size':19});
```

Find the most frequently occuring words in the corpus
```python
countList = lar.count(docsDf.DocText, returnRange=5000)
print(countList[:20])
```

Generate collocations (bigram or trigram) around a specific keyword, and store the collocations (and their scores as LL or PMI) in a new dataframe
```python
bigrams_FOX = lar.collocate(docsDf[f['FOX']], keyword='economy')
bigrams_CNN = lar.collocate(docsDf[f['CNN']], keyword='economy')

fox = lar.collatedTable(bigrams_FOX, scoreType='LL')
cnn = lar.collatedTable(bigrams_CNN, scoreType='LL')

```

Compare two collocation tables to see how different sources use the same word differently. 
```python
print(lar.compare(fox, cnn, 'Fox', 'CNN', retainKeyword='economy', print=False, unique=False))
```
