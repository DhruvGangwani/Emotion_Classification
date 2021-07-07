from flask import Flask, request
import json
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
from transformers import TFRobertaModel
roberta_model = TFRobertaModel.from_pretrained('roberta-base')
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import contractions
import re
import pandas as pd
import json



#custom 
from utils.preprocessing import preprocess
from utils.model import roberta_inference_encode, create_model

def inference(text_sentence, max_len, checkpoint_path):
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(preprocessed_text, maximum_length = max_len)
    model = create_model(roberta_model, max_len)
    model.load_weights(checkpoint_path)
    result = model.predict([input_ids, attention_masks])
    classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    # result = pd.DataFrame(dict(zip(classes, [round(x*100, 2)for x in result[0]])).items(), columns = ['Category', 'Confidence'])
    # plot_result(result)
    r = dict(zip(classes, [round(x*100, 2) for x in result[0]]))
    return r

app = Flask(__name__)

@app.route('/get_emotion', methods=["GET", "POST"])
def get_emotions():
	input_data = request.data
	data = json.loads(input_data)
	text = data['text']
	checkpoint_file = "/home/dhruv/nlp-suits/app/static/models/RoBERTa-Emotion/my_checkpoint"
	result = inference(text,  43, checkpoint_file)
	return result


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)