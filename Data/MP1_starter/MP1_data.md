# Machine Problem #1 

## Data format  

Data files are text files with the following format:

```

D N1 N2 N3

f_1_1_value f_1_2_value ... f_1_D_value

f_2_1_value f_2_2_value ... f_2_D_value

...

f_N1_1_value f_N1_2_value ... f_N1_D_value

f_1_1_value f_1_2_value ... f_1_D_value

f_2_1_value f_2_2_value ... f_2_D_value

...

f_N2_1_value f_N2_2_value ... f_N2_D_value

f_1_1_value f_1_2_value ... f_1_D_value

f_2_1_value f_2_2_value ... f_2_D_value

...

f_N3_1_value f_N3_2_value ... f_N3_D_value

```

where 

* D: integer,  the dimensionality of the data (i.e., the number of features in each instance)
* N1, N2, and N3: integer, the number of examples in class A, B, and C, respectively. 
* f_i_j_value: real number, the j-th feature value of the i-th data example

In the list of feature vectors, all of class A is listed first, then all of class B, then all of class C.

## Training & Testing Files

The training data is given in a file named **training#.txt**, and the testing data is given in a file named **testing#.txt**. Here # indicates the id of the training set.

We publish **training1.txt** and **testing1.txt** as the reference and students can use them for implementation. However, for final grading, we will use extra private training-testing pairs (**training2.txt**, **testing2.txt**, **training3.txt**, **testing3.txt**). These data files will have different example sizes and feature sizes (but all of them are **three-class** classification). So make use your code can work with **any feature dimensionality** and with **any number of training and testing instances** (within memory limitations)