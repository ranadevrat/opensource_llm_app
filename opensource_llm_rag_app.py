from flask import Flask, request, render_template, jsonify
import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
#from gpt4all import GPT4All
#from langchain.llms import GPT4All
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
# sentence transformers
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()


app = Flask(__name__)

# Set up LLM and embedding models
llm = Ollama(model="llama3.1:latest", request_timeout=120.0)


def build_sentence_window_index(documents,llm, embed_model=OpenAIEmbedding(),sentence_window_size=3,
                                save_dir="sentence_index",):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, embed_model = embed_model
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            embed_model = embed_model
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

from llama_index.llms.openai import OpenAI
from data.dataprovider import key
from llama_index.core import SimpleDirectoryReader
#OpenAI.api_key =  key

documents = SimpleDirectoryReader(
    input_files=[r"data/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

index = build_sentence_window_index(
    [document],
    #llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key=key),    
    llm = llm, #Replace this path with your model path
    save_dir="./sentence_index",
)

query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)

def chat_bot_rag(query):
    window_response = query_engine.query(
        query
    )

    return window_response



# Define your Flask routes
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']    
    bot_message = chat_bot_rag(user_message)    
    return jsonify({'response': str(bot_message)})

if __name__ == '__main__':
    app.run()
    #app.run(debug=True)
