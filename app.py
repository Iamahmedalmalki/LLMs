from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone, PodSpec
import openai
from urllib.parse import urljoin, urldefrag
from config import pinecone_api_key, openai_api_key, replicate_api_key

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Static URL for scraping
STATIC_URL = "https://u.ae/en/information-and-services"

pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists before creating it
index_name = 'llms-index'
if index_name not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of 'all-MiniLM-L6-v2' model embeddings
        metric='cosine',
        spec=PodSpec(
            environment='gcp-starter'
        )
    )
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')
openai.api_key = openai_api_key 

headers = {"Authorization": f"Token {replicate_api_key}"}

# Crawling Function
def crawl(url, base_url, visited=None):
    if visited is None:
        visited = set()
    
    # Normalize URL by removing fragment
    url = urldefrag(url).url
    
    # Check if the URL has already been visited
    if url in visited:
        return
    
    # Add the URL to the visited set
    visited.add(url)
    
    # Get the content of the URL, following redirections
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        
        # After redirections, check if the final URL is still within the base path
        final_url = urldefrag(response.url).url
        if not final_url.startswith(base_url):
            return
        
        # Check Content-Type to ensure it's HTML
        if 'text/html' not in response.headers.get('Content-Type', '').lower():
            return
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Store the raw HTML content
    store_in_vector_db([soup.get_text(separator=' ')])

    # Emit real-time updates
    socketio.emit('scrape_update', {'url': final_url, 'status': 'stored'})
    
    # Find all links in the page
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Resolve relative URLs
        next_url = urljoin(final_url, href)
        next_url = urldefrag(next_url).url  # Normalize next URL
        
        # Ensure the next URL is a subpath of the base URL
        if next_url.startswith(base_url):
            crawl(next_url, base_url, visited)
    
    print(f"Visited: {final_url}")

# Vectorization and Storage Functions
def vectorize_data(data):
    vectors = []
    for idx, text in enumerate(data):
        vector = model.encode(text)
        vectors.append((str(idx), vector))
    return vectors

def store_in_vector_db(data):
    vectors = vectorize_data(data)
    index.upsert(vectors)

def store_cleaned_text(url):
    visited = set()
    crawl(url, url, visited)

def query_vector_db(prompt, top_k=5):
    prompt_vector = model.encode(prompt)
    results = index.query([prompt_vector], top_k=top_k)
    relevant_texts = [match['metadata']['text'] for match in results['matches']]
    reference = " ".join(relevant_texts)
    keywords = reference.split()
    return reference, keywords

# LLM Interaction Functions
def get_llm_responses(reference, prompt):
    combined_prompt = f"Based only on the following search results, answer the user's query. Do not use any external knowledge:\n\n{reference}\n\nUser's query: {prompt}"
    responses = {}
    
    # GPT-4
    responses['gpt-4'] = openai.Completion.create(
        engine='gpt-4',
        prompt=combined_prompt,
        max_tokens=150
    ).choices[0].text

    # GPT-3.5-turbo
    responses['gpt-3.5-turbo'] = openai.Completion.create(
        engine='gpt-3.5-turbo',
        prompt=combined_prompt,
        max_tokens=150
    ).choices[0].text

    # Llama-2-70b-chat
    replicate_url = "https://api.replicate.com/v1/predictions"
    llama_payload = {
        "version": "replicate/llama-2-70b-chat",
        "input": {"prompt": combined_prompt}
    }
    llama_response = requests.post(replicate_url, json=llama_payload, headers=headers)
    responses['llama-2-70b-chat'] = llama_response.json()['output']

    # Falcon-40b-instruct
    falcon_payload = {
        "version": "joehoover/falcon-40b-instruct",
        "input": {"prompt": combined_prompt}
    }
    falcon_response = requests.post(replicate_url, json=falcon_payload, headers=headers)
    responses['falcon-40b-instruct'] = falcon_response.json()['output']

    return responses

# Evaluation Functions
def compute_similarity(response, reference):
    response_embedding = model.encode(response)
    reference_embedding = model.encode(reference)
    similarity = util.cos_sim(response_embedding, reference_embedding)
    return similarity.item()

def keyword_matching(response, keywords):
    matches = [keyword for keyword in keywords if keyword in response]
    return len(matches) / len(keywords)

def aggregate_scores(similarity, keywords_score):
    final_score = (
        similarity * 0.7 + 
        keywords_score * 0.3
    )
    return final_score

def evaluate_responses(responses, reference, keywords):
    scores = {}
    for model, response in responses.items():
        similarity = compute_similarity(response, reference)
        keywords_score = keyword_matching(response, keywords)
        
        final_score = aggregate_scores(similarity, keywords_score)
        scores[model] = final_score

    best_model = max(scores, key=scores.get)
    return best_model, scores

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_prompt = request.form.get('prompt')
    reference, keywords = query_vector_db(user_prompt)
    responses = get_llm_responses(reference, user_prompt)
    best_model, scores = evaluate_responses(responses, reference, keywords)
    return jsonify({'best_model': best_model, 'scores': scores})

@app.route('/scrape', methods=['POST'])
def scrape():
    store_cleaned_text(STATIC_URL)
    return jsonify({'status': 'success'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '_main_':
    socketio.run(app, debug=True)