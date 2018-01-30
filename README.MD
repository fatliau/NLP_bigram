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
JCMacBook:CS6320_NLP JC$ python homework1.py 'corpus.txt' "Apple computer is the first product of the company" "Apple introduced the new version of iPhone in 2008"

bigram counts table for Sentence 1
          Apple  computer  is  the  first  product  of  company
Apple         0         0   0   23      2        0  17        0
computer      0         0   0    0      1        0   2        0
is            7         0   0    0      0        0   0        0
the           1         0   5    0      0        0  85        0
first         0         0   0   28      0        0   0        0
product       0         0   0    1      1        0   0        0
of            0         0   0    0      1        0   0        0
company       0         0   0   43      0        0   2        0

bigram counts table for Sentence 2
            Apple  introduced  the  new  version  of  iPhone  in  2008
Apple           0           1   23    1        0  17       0   8     0
introduced     20           0    0    0        0   0       0   0     0
the             1          15    0    0        0  85       0  55     0
new             0           2    6    0        0   0       0   0     0
version         0           0    0    0        0   0       0   0     0
of              0           0    0    0        3   0       0   0     0
iPhone          0           0   29    1        0   1       0   1     0
in              6           0    0    0        0   0       0   0     0
2008            0           0    0    0        0   0       0   1     0

bigram probability table for Sentence 1
             Apple  computer        is       the     first  product        of  company
Apple     0.000000       0.0  0.000000  0.051111  0.004444      0.0  0.037778      0.0
computer  0.000000       0.0  0.000000  0.000000  0.058824      0.0  0.117647      0.0
is        0.129630       0.0  0.000000  0.000000  0.000000      0.0  0.000000      0.0
the       0.001326       0.0  0.006631  0.000000  0.000000      0.0  0.112732      0.0
first     0.000000       0.0  0.000000  0.571429  0.000000      0.0  0.000000      0.0
product   0.000000       0.0  0.000000  0.032258  0.032258      0.0  0.000000      0.0
of        0.000000       0.0  0.000000  0.000000  0.002674      0.0  0.000000      0.0
company   0.000000       0.0  0.000000  0.551282  0.000000      0.0  0.025641      0.0

bigram probability table for Sentence 2
               Apple  introduced       the       new   version        of  iPhone        in  2008
Apple       0.000000    0.002222  0.051111  0.002222  0.000000  0.037778     0.0  0.017778   0.0
introduced  0.526316    0.000000  0.000000  0.000000  0.000000  0.000000     0.0  0.000000   0.0
the         0.001326    0.019894  0.000000  0.000000  0.000000  0.112732     0.0  0.072944   0.0
new         0.000000    0.041667  0.125000  0.000000  0.000000  0.000000     0.0  0.000000   0.0
version     0.000000    0.000000  0.000000  0.000000  0.000000  0.000000     0.0  0.000000   0.0
of          0.000000    0.000000  0.000000  0.000000  0.008021  0.000000     0.0  0.000000   0.0
iPhone      0.000000    0.000000  0.475410  0.016393  0.000000  0.016393     0.0  0.016393   0.0
in          0.020548    0.000000  0.000000  0.000000  0.000000  0.000000     0.0  0.000000   0.0
2008        0.000000    0.000000  0.000000  0.000000  0.000000  0.000000     0.0  0.090909   0.0

bigram counts table for Sentence 1 with smoothing
          Apple  computer  is  the  first  product  of  company
Apple         1         1   1   24      3        1  18        1
computer      1         1   1    1      2        1   3        1
is            8         1   1    1      1        1   1        1
the           2         1   6    1      1        1  86        1
first         1         1   1   29      1        1   1        1
product       1         1   1    2      2        1   1        1
of            1         1   1    1      2        1   1        1
company       1         1   1   44      1        1   3        1

bigram counts table for Sentence 2 with smoothing
            Apple  introduced  the  new  version  of  iPhone  in  2008
Apple           1           2   24    2        1  18       1   9     1
introduced     21           1    1    1        1   1       1   1     1
the             2          16    1    1        1  86       1  56     1
new             1           3    7    1        1   1       1   1     1
version         1           1    1    1        1   1       1   1     1
of              1           1    1    1        4   1       1   1     1
iPhone          1           1   30    2        1   2       1   2     1
in              7           1    1    1        1   1       1   1     1
2008            1           1    1    1        1   1       1   2     1

bigram probability table for Sentence 1 with smoothing
             Apple  computer        is       the     first   product        of   company
Apple     0.000259  0.000259  0.000259  0.006213  0.000777  0.000259  0.004660  0.000259
computer  0.000292  0.000292  0.000292  0.000292  0.000583  0.000292  0.000875  0.000292
is        0.002307  0.000288  0.000288  0.000288  0.000288  0.000288  0.000288  0.000288
the       0.000480  0.000240  0.001440  0.000240  0.000240  0.000240  0.020638  0.000240
first     0.000289  0.000289  0.000289  0.008377  0.000289  0.000289  0.000289  0.000289
product   0.000290  0.000290  0.000290  0.000581  0.000581  0.000290  0.000290  0.000290
of        0.000264  0.000264  0.000264  0.000264  0.000528  0.000264  0.000264  0.000264
company   0.000286  0.000286  0.000286  0.012604  0.000286  0.000286  0.000859  0.000286

bigram probability table for Sentence 2 with smoothing
               Apple  introduced       the       new   version        of    iPhone        in      2008
Apple       0.000259    0.000518  0.006213  0.000518  0.000259  0.004660  0.000259  0.002330  0.000259
introduced  0.006085    0.000290  0.000290  0.000290  0.000290  0.000290  0.000290  0.000290  0.000290
the         0.000480    0.003840  0.000240  0.000240  0.000240  0.020638  0.000240  0.013439  0.000240
new         0.000289    0.000867  0.002023  0.000289  0.000289  0.000289  0.000289  0.000289  0.000289
version     0.000292    0.000292  0.000292  0.000292  0.000292  0.000292  0.000292  0.000292  0.000292
of          0.000264    0.000264  0.000264  0.000264  0.001056  0.000264  0.000264  0.000264  0.000264
iPhone      0.000288    0.000288  0.008636  0.000576  0.000288  0.000576  0.000288  0.000576  0.000288
in          0.001889    0.000270  0.000270  0.000270  0.000270  0.000270  0.000270  0.000270  0.000270
2008        0.000292    0.000292  0.000292  0.000292  0.000292  0.000292  0.000292  0.000584  0.000292

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