import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Ensure nltk resources are available
nltk.download('punkt')

# Sample text corpus
corpus = """The quick brown fox jumps over the lazy dog. The dog barks back at the fox.
            The fox runs away into the forest. A quick red fox appears."""

# Tokenize the text
tokens = word_tokenize(corpus.lower())

# 1. Unigrams
unigrams = tokens
unigram_counts = Counter(unigrams)

# 2. Bigrams
bigrams = list(ngrams(tokens, 2))
bigram_counts = Counter(bigrams)

# 3. Trigrams
trigrams = list(ngrams(tokens, 3))
trigram_counts = Counter(trigrams)

# 4. Bigram Probabilities
bigram_probabilities = defaultdict(lambda: defaultdict(float))
for (w1, w2), count in bigram_counts.items():
    bigram_probabilities[w1][w2] = count / unigram_counts[w1]

# 5. Next Word Prediction
def predict_next_word(word):
    possible_words = bigram_probabilities[word]
    if not possible_words:
        return None
    next_word = max(possible_words, key=possible_words.get)
    return next_word

# Writing the output to a file
output_file = "ngram_output.txt"

with open(output_file, "w") as file:
    # Unigrams
    file.write("Unigrams and their frequencies:\n")
    for word, count in unigram_counts.items():
        file.write(f"{word}: {count}\n")
    
    # Bigrams
    file.write("\nBigrams and their frequencies:\n")
    for bigram, count in bigram_counts.items():
        file.write(f"{bigram}: {count}\n")
    
    # Trigrams
    file.write("\nTrigrams and their frequencies:\n")
    for trigram, count in trigram_counts.items():
        file.write(f"{trigram}: {count}\n")
    
    # Bigram Probabilities
    file.write("\nBigram Probabilities:\n")
    for w1, w2_probs in bigram_probabilities.items():
        for w2, prob in w2_probs.items():
            file.write(f"P({w2} | {w1}) = {prob:.2f}\n")

    # Next word prediction examples
    test_words = ['the', 'quick', 'fox']
    file.write("\nNext Word Predictions:\n")
    for word in test_words:
        next_word = predict_next_word(word)
        file.write(f"Next word after '{word}': {next_word}\n")

print(f"Output written to {output_file}")

# Plotting the graphs
def plot_frequency_graph(data, title, xlabel, ylabel, output_filename):
    labels, frequencies = zip(*data)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, frequencies, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"{title} graph saved as {output_filename}")

# Plot Unigram Frequencies
unigram_data = unigram_counts.most_common(10) # Top 10 unigrams
plot_frequency_graph(unigram_data, 'Top 10 Unigram Frequencies', 'Unigrams', 'Frequency', 'unigram_frequencies.png')

# Plot Bigram Frequencies
bigram_data = bigram_counts.most_common(10) # Top 10 bigrams
bigram_labels = [f"{w1} {w2}" for (w1, w2) in [pair for pair, _ in bigram_data]]
bigram_frequencies = [count for _, count in bigram_data]
plot_frequency_graph(list(zip(bigram_labels, bigram_frequencies)), 'Top 10 Bigram Frequencies', 'Bigrams', 'Frequency', 'bigram_frequencies.png')

# Plot Trigram Frequencies
trigram_data = trigram_counts.most_common(10) # Top 10 trigrams
trigram_labels = [f"{w1} {w2} {w3}" for (w1, w2, w3) in [triplet for triplet, _ in trigram_data]]
trigram_frequencies = [count for _, count in trigram_data]
plot_frequency_graph(list(zip(trigram_labels, trigram_frequencies)), 'Top 10 Trigram Frequencies', 'Trigrams', 'Frequency', 'trigram_frequencies.png')
