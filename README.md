Here,I have tried to classify whether a person has a _benign_ or a _malignant_ type of cancer depending upon certain relavant features.The following steps were done in this code:

1. After loading the data, some statistical details were known using pandas.
2. Data visualization was done b/w independant features and the dependant feature.
3. A heatmap was used to see the most correlated features.
4. After splitting the data into training and test set, data normalization was done on all the independent features.
5. Then a neural model was built with 2 hidden layers using the _relu activation function_ and _sigmoid activation function_ in the output layer. Along with that _BinaryCrossEntropy_ was used as the loss function and _Adam_ optimzer was used for compiling the model.
6.  To reduce the validation loss, **Early Stopping** method was used, which reduced the number of epochs from 600 to only 92.
7.  Finally model performance was checked by building a _classification report_ and the accuracy of the model was 96%.

The dataset has been attached. Thanks!!
