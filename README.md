# CMSC5707 Assigment 1

Student Name: Lau Ying Yeung, Eddy

Student ID: 1155093989

### For running the program,
Firstly, make sure to install or use Python 3.8.12 for running this program.

You can validate your python version by below commands, depends on your machine Python environment settings.
```sh
python -V
```
```sh
python3 -V
```
```sh
python3.8 -V
```


Secondly, create virtualenv for installing the libraries stated in requirements.txt
```sh
python3.8 -m venv env
```
**make sure to run this command at the root level of the project directory**

If venv is not installed with your Python version, please install venv before running the create environment command.

Thirdly, activate the virtual environment and install the libraries according to the requirements.txt by running the below command.
```sh
source env/bin/activate
```
**make sure your directory location is in the root level of the project**
```sh
pip install -r requirements.txt
```

At last, run the hw1.py script using Python
```sh
python hw1.py
```

### How to examine the program result

#### **For question 1 and 2**
The recordings are stored in the recordings folder. The language for recording the number of my student id is english. And the recordings are using 48kHz sample rate and 16 bit resolution.

#### **For question 3**
The hw1.py will run the function ```answer_q3``` for plotting the signals from set A recordings and save the graphs to a folder called 'plot' one-by-one. 

#### **For question 4**
The hw1.py will run the function ```answer_q4```. It will first show the target filename, frame rate and total frames of the signal.

In section A, it will start calculate the end point detection part. The result will be shown in the console, and it consists of two rows. The first row shows the end point start time and end time in ```frame```, while the second row shows the end point start time and end time in ```seconds```. Also, a graph will be shown and saved for illustrating the end point detection result for visualizing the data easier.

In section B, the segment will be cutted between the start time and end time it found in section A. The extract method for getting the segment is [start_time, start_time + 20ms].

In section C, a graph for the Fourier Transform will be shown and saved.

In section D, a graph for the pre-emphasis signal will be shown and saved.

In section E, it will calculate the LPC paramters, and the result will be shown in the console.

Console Log Example: 
```
Target: s1A.wav
framerate: 48000
Total frames: 38424
End Point: start_frame = 5 , end_frame = 46
End Point: start_time = 0.05, end_time = 0.46
LPC parameters: [ 0.29030567  0.0535346   0.42777176  0.16377339  0.15895562  0.09668412
 -0.0675887  -0.08788707  0.01839802 -0.08472624]
```

#### **For question 5**
The hw1.py will run the function ```answer_q5```. It will first show the target filename, frame rate and total frames of the signal.

In section A, it will find the MFCCs paramters for set A and set B recordings and save as 'mfcc_paramter_x.csv' where x is the record name for examining usage.

In section B, it selects s1A and s1B for finding the minimum accumulated score. The result will be shown in the console. And for comparing the result, it also compare s1A with s8B just to illustrate the difference of the minimum accumulated score.

In section C, a graph of the confusion matrix will be shown and saved as 'confusion-matrix.jpg' and 'confusion-matrix.csv' for examining usage.

In section D, a graph for the optimal path will be shown and saved as 'x1-to-x2-accumulation-matrix.jpg' where x1 is the actual class and x2 is the predicted class for examining usage.

Console Log Example:
```
Minimum Accumulated Score for s1A to s1B = 3111.298514150166
Minimum Accumulated Score for s1A to s8B = 5034.927635586084 (for comparison)
```