# Unraveling-Sentiments-in-Tweets-Using-LSTM-based-Deep-Learning

Team members:

Nikhil Baghel (21BDS044)
Shubham Kumar (21BDS063)
Sidharth Kaushik (21BDS064)
Yuvraaj Bhama (21BDS071)

There is a main file which contains all predefined functions like nltk, pandas, matplotlib, keras. We have trained the lstm model. There is a predict function def predict(text, include_neutral=True). It is taking text as input. We can put any type of input like numbers, special symbols, alphabets and so on. So the user is expected to give the input in the form of string and the model will classify is positive or negative with some score. If the score is greater than 0.5, it means that the text written is positive in nature and if it is less than 0.5, it means that it is negative in nature. It is tokenizing the text using pad sequences method, decoding the text and giving the output in the form of label, score and elapsed time.
