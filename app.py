import json
from flask import Flask, jsonify, make_response, abort, request
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydantic import BaseModel, parse
from pydantic_webargs import webargs
from urllib.parse import urlparse, parse_qs
import urllib.parse
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)

# ... other functions (refer to the original code)

@app.route('/summarize/check', methods=['GET'])
def transcript(video_id):
    try:
        Transc = YouTubeTranscriptApi.get_transcript(video_id)
        return jsonify(Transc)
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return abort(400, description="Failed to get transcript from YouTube.")

@app.route('/summarize/summary', methods=['GET'])
def summary(string):
    try:
        # Initialize the T5 model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

        # Encode the text into tensor of integers
        inputs = tokenizer.encode("summarize: " + string, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summarization output
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2, num_return_sequences=1, early_stopping=True)

        # Decode the output and return the summary
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return abort(500, description="Failed to summarize text.")

# ... other functions (refer to the original code)

if __name__ == '__main__':
    app.run(debug=True)