# NLP automatic speech recognition - bigram model

## what’s this
This is my [homework 1 from CS6320 in the University of Texas at Dallas, Spring 2018](http://www.hlt.utdallas.edu/~moldovan/CS6320.18S/Homework1.pdf)

## set up
1. environment: Python 3
2. package used: nltk, pandas
3. put all files in the same folder: homework1.py, corpus.txt(or any .txt as the word training set)

## execute command
python homework1.py 'corpus.txt' "Apple computer is the first product of the company" "Apple introduced the new version of iPhone in 2008"

## assumption
1. case sensitive, ex:”iPhone” != “iphone”. There in Sentence 2 was an “iphone”, I change it to “iPhone” for a better result. 
2. The set size to be compared is sentence length minus one, for the reason to simply the calculation, so I ignore the start sign(denoted as ‘^’) and end sign(denoted as ‘$’)
3. When bigram probability without smoothing, if the word of sentence never show up in the training set, to prevent dividing by zero, I skip this bigram and list how many bigrams are skipped by this case.

## result example
```
JCMacBook:CS6320_NLP JC$ python homework1.py 'corpus.txt' "Apple computer is the first product of the company" "Apple introduced the new version of iPhone in 2008"
```
```
bigram counts table for Sentence 1
          Apple  computer  is  the  first  product  of  company
Apple         0         0   7    1      0        0   0        0
computer      0         0   0    0      0        0   0        0
is            0         0   0    5      0        0   0        0
the          23         0   0    0     28        1   0       43
first         2         1   0    0      0        1   1        0
product       0         0   0    0      0        0   0        0
of           17         2   0   85      0        0   0        2
company       0         0   0    0      0        0   0        0

bigram counts table for Sentence 2
            Apple  introduced  the  new  version  of  iPhone  in  2008
Apple           0          20    1    0        0   0       0   6     0
introduced      1           0   15    2        0   0       0   0     0
the            23           0    0    6        0   0      29   0     0
new             1           0    0    0        0   0       1   0     0
version         0           0    0    0        0   3       0   0     0
of             17           0   85    0        0   0       1   0     0
iPhone          0           0    0    0        0   0       0   0     0
in              8           0   55    0        0   0       1   0     1
2008            0           0    0    0        0   0       0   0     0

bigram probability table for Sentence 1
             Apple  computer       is       the     first   product        of   company
Apple     0.000000  0.000000  0.12963  0.001326  0.000000  0.000000  0.000000  0.000000
computer  0.000000  0.000000  0.00000  0.000000  0.000000  0.000000  0.000000  0.000000
is        0.000000  0.000000  0.00000  0.006631  0.000000  0.000000  0.000000  0.000000
the       0.051111  0.000000  0.00000  0.000000  0.571429  0.032258  0.000000  0.551282
first     0.004444  0.058824  0.00000  0.000000  0.000000  0.032258  0.002674  0.000000
product   0.000000  0.000000  0.00000  0.000000  0.000000  0.000000  0.000000  0.000000
of        0.037778  0.117647  0.00000  0.112732  0.000000  0.000000  0.000000  0.025641
company   0.000000  0.000000  0.00000  0.000000  0.000000  0.000000  0.000000  0.000000

bigram probability table for Sentence 2
               Apple  introduced       the       new  version        of    iPhone        in      2008
Apple       0.000000    0.526316  0.001326  0.000000      0.0  0.000000  0.000000  0.020548  0.000000
introduced  0.002222    0.000000  0.019894  0.041667      0.0  0.000000  0.000000  0.000000  0.000000
the         0.051111    0.000000  0.000000  0.125000      0.0  0.000000  0.475410  0.000000  0.000000
new         0.002222    0.000000  0.000000  0.000000      0.0  0.000000  0.016393  0.000000  0.000000
version     0.000000    0.000000  0.000000  0.000000      0.0  0.008021  0.000000  0.000000  0.000000
of          0.037778    0.000000  0.112732  0.000000      0.0  0.000000  0.016393  0.000000  0.000000
iPhone      0.000000    0.000000  0.000000  0.000000      0.0  0.000000  0.000000  0.000000  0.000000
in          0.017778    0.000000  0.072944  0.000000      0.0  0.000000  0.016393  0.000000  0.090909
2008        0.000000    0.000000  0.000000  0.000000      0.0  0.000000  0.000000  0.000000  0.000000

bigram counts table for Sentence 1 with smoothing
          Apple  computer  is  the  first  product  of  company
Apple         1         1   8    2      1        1   1        1
computer      1         1   1    1      1        1   1        1
is            1         1   1    6      1        1   1        1
the          24         1   1    1     29        2   1       44
first         3         2   1    1      1        2   2        1
product       1         1   1    1      1        1   1        1
of           18         3   1   86      1        1   1        3
company       1         1   1    1      1        1   1        1

bigram counts table for Sentence 2 with smoothing
            Apple  introduced  the  new  version  of  iPhone  in  2008
Apple           1          21    2    1        1   1       1   7     1
introduced      2           1   16    3        1   1       1   1     1
the            24           1    1    7        1   1      30   1     1
new             2           1    1    1        1   1       2   1     1
version         1           1    1    1        1   4       1   1     1
of             18           1   86    1        1   1       2   1     1
iPhone          1           1    1    1        1   1       1   1     1
in              9           1   56    1        1   1       2   1     2
2008            1           1    1    1        1   1       1   1     1

bigram probability table for Sentence 1 with smoothing
             Apple  computer        is       the     first   product        of   company
Apple     0.000259  0.000292  0.002307  0.000480  0.000289  0.000290  0.000264  0.000286
computer  0.000259  0.000292  0.000288  0.000240  0.000289  0.000290  0.000264  0.000286
is        0.000259  0.000292  0.000288  0.001440  0.000289  0.000290  0.000264  0.000286
the       0.006213  0.000292  0.000288  0.000240  0.008377  0.000581  0.000264  0.012604
first     0.000777  0.000583  0.000288  0.000240  0.000289  0.000581  0.000528  0.000286
product   0.000259  0.000292  0.000288  0.000240  0.000289  0.000290  0.000264  0.000286
of        0.004660  0.000875  0.000288  0.020638  0.000289  0.000290  0.000264  0.000859
company   0.000259  0.000292  0.000288  0.000240  0.000289  0.000290  0.000264  0.000286

bigram probability table for Sentence 2 with smoothing
               Apple  introduced       the       new   version        of    iPhone        in      2008
Apple       0.000259    0.006085  0.000480  0.000289  0.000292  0.000264  0.000288  0.001889  0.000292
introduced  0.000518    0.000290  0.003840  0.000867  0.000292  0.000264  0.000288  0.000270  0.000292
the         0.006213    0.000290  0.000240  0.002023  0.000292  0.000264  0.008636  0.000270  0.000292
new         0.000518    0.000290  0.000240  0.000289  0.000292  0.000264  0.000576  0.000270  0.000292
version     0.000259    0.000290  0.000240  0.000289  0.000292  0.001056  0.000288  0.000270  0.000292
of          0.004660    0.000290  0.020638  0.000289  0.000292  0.000264  0.000576  0.000270  0.000292
iPhone      0.000259    0.000290  0.000240  0.000289  0.000292  0.000264  0.000288  0.000270  0.000292
in          0.002330    0.000290  0.013439  0.000289  0.000292  0.000264  0.000576  0.000270  0.000584
2008        0.000259    0.000290  0.000240  0.000289  0.000292  0.000264  0.000288  0.000270  0.000292

the total probabilities for Sentence 1
did multiplied:  8  compare length:  8
ignored  0  probabilities due to zero frequency
0.0

the total probabilities for Sentence 2
did multiplied:  8  compare length:  8
ignored  0  probabilities due to zero frequency
0.0

the total probabilities for Sentence 1 with smoothing
did multiplied:  8  compare length:  8
4.045759371642444e-23

the total probabilities for Sentence 2 with smoothing
did multiplied:  8  compare length:  8
1.323528368268833e-24

Sentence 1 is better, which is " Apple computer is the first product of the company "
```
