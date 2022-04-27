# AsianReligionsClassification

An experiment project for Pattern Recognition which is a course released by Yong Wang from Chinese Academy of Science.
Considering not very much time spent on this task, it may be naive and not bug-free.
If you find any problems with it, feel free to send an email to buptmsg@gmail.com.

The python script analysis.py implements a simple but efficient method for classification.
And also it contains the code for classfication models based on scikit-learn as a contrast.

## Result

|Method	|Avg/stddev of acc|
|-----|------|
|Linear Regression|	37.39 / 4.46 |
|Random Forest	|83.45 / 4.13 |
|Decision Tree	|69.85 / 4.35 |
|SVC	| 77.00 / 4.16 |
|高频分析|	95.40 / 1.57 |
|低频分析	| 95.76 / 1.78 |
|高频+低频分析|	95.40 / 1.69 |
|__词汇去除__	| __95.84 / 1.65__ |
