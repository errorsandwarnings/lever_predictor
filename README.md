# lever_predictor
This project attempts at using an LSTM to predict the salary using unstructured textual representation of a description posted at lever urls.

# Install Requirements
	pip install -r requirements.txt

# Download fasttext model and place it in ./fasttext 
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
	unzip wiki-news-300d-1M.vec.zip

# Ensure you download spacy en_sm model using spacy module
	python -m spacy download en_core_web_sm

# Run the server
  python web.py
  
# Open url 
  wget http://localhost:5000/salary/predict/<url>
	
# Wanna Retrain yourself ? Follow the instructions
  python train_model/training.py

# Copy your model at the base directory.

# Enjoy !
