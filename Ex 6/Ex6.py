import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

max_words=5000
max_len=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_words)
X_train=pad_sequences(x_train,maxlen=max_len)
X_test=pad_sequences(x_test,maxlen=max_len)

model=Sequential()
model.add(Embedding(input_dim=max_words,output_dim=32,input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print("Training.....")
model.fit(X_train,y_train,epochs=2,batch_size=64,validation_split=0.2)
loss,acc=model.evaluate(X_test,y_test)
print(f"\nTest Accuracy:{acc:0.4f}")
word_index=imdb.get_word_index()
reverse_word_index={v:k for(k,v) in word_index.items()}


def decode_review(review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in review])
sample_review=X_test[0]
prediction=model.predict(sample_review.reshape(1,-1))[0][0]
print("\nReview Text:",decode_review(x_test[0]))
print("Predicted Sentiment:","Positive" if prediction>0.5 else "Negative")
