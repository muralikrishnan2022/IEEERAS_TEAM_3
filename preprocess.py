import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# To load a document
'''
file_path = input("Enter file path:")
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read().replace("\n", "")
'''
# Sample text 
text = """You probably already know that it is important to have a king-size breakfast every morning. do you know why Your body is hungry in the morning because you haven’t eaten for about 8-10 hours? Breakfast is therefore the first meal of the day, and therefore, the most important. Imagine driving without fuel; This is exactly how your body feels without fuel from a nutritious breakfast. Nowadays many people skip breakfast to lose weight. Nutritionists are alarmed by this trend, as it is mandatory to eat breakfast within two hours of waking up. Depriving the body of energy can lead to serious health problems in the long run. Forget silly celebrities and their absurd ways to lose weight. Never miss breakfast!"""

# Sentence Tokenization
sentences = sent_tokenize(text)

# Word Tokenization & Lowercasing
words = [word_tokenize(sent.lower()) for sent in sentences]

# Remove Stopwords & Punctuation
stop_words = set(stopwords.words("english"))
cleaned_words = [[word for word in sent if word not in stop_words and word not in string.punctuation] for sent in words]

pos_tagged_sentences = [pos_tag(sent) for sent in cleaned_words]

print("Original Sentences:", sentences)
print("\nTokenized Words:", words)
print("\nCleaned Words (No Stopwords/Punctuation):", cleaned_words)
print("\nPOS Tagged Sentences:", pos_tagged_sentences)