from Hybrid_Dual_Indexing import Keyword_Search, Semantic_Search
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import joblib
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Vector_DB:

    def __init__(self, semantic_search_model, what_to_do = "CREATE_DATABASE"):

        if what_to_do == "LOAD_DATABASE":
            self.load_vector_database (semantic_search_model.embeddings_model)

        elif what_to_do == "CREATE_DATABASE":

            number_of_neighbors = 32
            embedding_dimension = len (semantic_search_model.embeddings_model.embed_query ("a"))
            
            index_HNSW_method = faiss.IndexHNSWFlat (embedding_dimension, number_of_neighbors, faiss.METRIC_INNER_PRODUCT)
            self.vector_database = FAISS (embedding_function = semantic_search_model.embeddings_model,
                                    index = index_HNSW_method,
                                    docstore = InMemoryDocstore (),
                                    index_to_docstore_id = {},
                                    normalize_L2 = True)

    def add_doc_to_vector_database (self, keyword_search_model):

        flatten_chunks = keyword_search_model.flatten_chunks
        self.vector_database.add_documents (flatten_chunks)

    def search (self, user_query, top_k = 20):

        top_k_chunks = self.vector_database.similarity_search (user_query, k = top_k)
        return top_k_chunks

    def save_vector_database (self, database_path = "FAISS_Database"):

        self.vector_database.save_local (database_path)
    
    def load_vector_database (self, embeddings_model, database_path = "FAISS_Database"):
        
        self.vector_database = FAISS.load_local (database_path, embeddings_model, allow_dangerous_deserialization = True)


if __name__ == "__main__":

    semantic_search_model = joblib.load ("Semantic_Model.pkl")
    keyword_search_model = joblib.load ("Keyword_Model.pkl")

    Vector_DB = Vector_DB (semantic_search_model)
    Vector_DB.add_doc_to_vector_database (keyword_search_model)
    Vector_DB.save_vector_database ()