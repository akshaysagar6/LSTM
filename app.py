import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model=load_model('model.h5')
with open('token.pickle','rb') as handle:
    token=pickle.load(handle)
 
def predict_next(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] #taking a subset of words if token_list length is greater than max_sequence_len 
        #token_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #max_sequence_len = 5 ,token_list[-(max_sequence_len-1):] will be token_list[-4:], which gives [7, 8, 9, 10].
    else:
        token_list = token_list
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predict_word_index = np.argmax(predicted, axis=1)[0]  # Extract the integer from the array
    
    for word, index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word
    
    return None

st.title('Next Word Predictor')
input_word=st.text_input('Enter the sequence of word')
if st.button('Predict next word'):
    max_seq=model.input_shape[1]+1
    next_word=predict_next(model,token,input_word,max_seq)
    st.write(input_word,next_word)