from flask import Flask, request, render_template, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the 20newsgroup dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the dataset to create the tf-idf matrix
tfidf_matrix = vectorizer.fit_transform(newsgroups.data)

# Define a function to search for documents based on a query string
def search_documents(query, n=10):
    # Convert the query string to a tf-idf vector
    query_vector = vectorizer.transform([query])

    # Calculate the cosine similarity between the query vector and all document vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort the document indices by their cosine similarity to the query vector
    document_indices = cosine_similarities.argsort()[::-1][:n]

    # Create a list of the top n most similar documents
    results = []
    for index in document_indices:
        title = newsgroups.target_names[newsgroups.target[index]]
        text = newsgroups.data[index]
        results.append((title, text))

    # Return the list of results
    return results

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the search functionality
@app.route('/search', methods=['POST'])
def search():
    try:
        # Get the query string from the form
        query = request.form['query']

        # Search for documents related to the query string
        results = search_documents(query)

        # Render the search results template with the results
        return render_template('results.html', results=results)

    except Exception as e:
        # Return an error message if an exception occurs
        error = 'An error occurred: {}'.format(str(e))
        return jsonify({'error': error}), 500

# Run the web application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
