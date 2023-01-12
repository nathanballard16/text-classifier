import pandas as pd
import os
import newspaper

infile = './data/gdelt_raw/India_UMV.csv'
outfile = './data/gdelt/India_UMV/'

print(infile)
oldData = pd.read_csv(infile, encoding="ISO-8859-1", engine='python')
print(oldData.shape)
urlList = oldData['SOURCEURL'].drop_duplicates()
print(urlList)

articleNumber = 0
for url in urlList:
    try:
        print(url)
        article = newspaper.Article(url)
        article.download()
        article.parse()
        print(str(articleNumber))
        file1 = open(outfile + str(articleNumber) + ".txt", "w+")
        file1.write(article.title + '\n\n')
        file1.write(article.text + '\n')
        file1.close()
        articleNumber += 1
    except Exception as e:
        print(e.__str__())
