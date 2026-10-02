from Hybrid_Dual_Indexing import Keyword_Search, Semantic_Search
from Knowledge_Graph import Knowledge_Graphbase
from Vector_Database import Vector_DB
from sentence_transformers import CrossEncoder
import torch
import joblib


class Retriever:

    def __init__ (self):

        # ONLY THIS PART IS GENERATED FROM GEMINI TO MOVE MY 'file.pkl' FROM MPS (MACBOOK) TO CPU (LINUX) FOR DOCKER
        # ------------------------------------------------------------------------------------------------------------
        original_storage_init = torch.UntypedStorage.__new__
        def patched_storage_new(cls, *args, **kwargs):
            if 'device' in kwargs and kwargs['device'] == 'mps':
                kwargs['device'] = 'cpu'
            return original_storage_init(cls, *args, **kwargs)
        
        torch.UntypedStorage.__new__ = patched_storage_new
        if not hasattr(torch.serialization, '_mps_deserialize'):
            torch.serialization._mps_deserialize = lambda obj, location: obj.cpu()
        try:
            self.semantic_search_model = joblib.load ("Semantic_Model.pkl")
            if hasattr (self.semantic_search_model, 'to'):
                self.semantic_search_model.to ('cpu')

        except Exception:
            with open ("Semantic_Model.pkl", "rb") as file:
                self.semantic_search_model = torch.load (
                    file, 
                    map_location='cpu', 
                    weights_only=False
            )
        torch.UntypedStorage.__new__ = original_storage_init
        # ------------------------------------------------------------------------------------------------------------

        self.rerank_model = CrossEncoder ("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.inverted_index = joblib.load ("Keyword_Model.pkl")
        self.vector_database = Vector_DB (self.semantic_search_model, "LOAD_DATABASE")
        self.knowledge_database = Knowledge_Graphbase ()
        self.knowledge_database.load_local ()

    def merge_multi_query_retrieval (self, multi_query_retrieval, decay_rank = 60, keep_top_k_chunk = 10):

        rank_docs = {}
        raw_docs_mapping = {}

        for each_query_retrieval in multi_query_retrieval:
            for rank, doc in enumerate (each_query_retrieval):

                doc_rank = 1 / (decay_rank + rank)
                doc_content = doc.page_content
                raw_docs_mapping[doc_content] = doc

                if doc_content in rank_docs:
                    rank_docs[doc_content] += doc_rank
                else:
                    rank_docs[doc_content] = doc_rank

        key = lambda item: item[1]
        rank_docs = sorted (rank_docs.items (), key = key, reverse = True)

        raw_docs = []
        keep_top_k_chunk = min (keep_top_k_chunk, len (rank_docs))

        for index in range (keep_top_k_chunk):

            doc_content = rank_docs[index][0]
            raw_doc = raw_docs_mapping[doc_content]
            raw_docs.append (raw_doc)
        
        return raw_docs

    def merge_hybrid_query_retrieval (self, rewrite_query, keyword_chunks, semantic_chunks, keep_top_k_chunk = 5):

        pairs = []
        merge_docs = []

        if keyword_chunks:
            for chunk in keyword_chunks:
                chunk_content = chunk.page_content
                pairs.append ([rewrite_query, chunk_content])
                merge_docs.append (chunk)

        if semantic_chunks:
            for chunk in semantic_chunks:
                chunk_content = chunk.page_content
                pairs.append ([rewrite_query, chunk_content])
                merge_docs.append (chunk)
        
        if not pairs:
            return []

        cls_scores = self.rerank_model.predict (pairs)
        document_combine_with_cls_score = zip (merge_docs, cls_scores)

        key = lambda pair: pair[1]
        zip_list_sort = sorted (document_combine_with_cls_score, key = key, reverse = True)
        
        raw_docs = []
        keep_top_k_chunk = min (keep_top_k_chunk, len (zip_list_sort))

        for index in range (keep_top_k_chunk):
            pair = zip_list_sort[index]
            doc = pair[0]
            raw_docs.append (doc)
        
        return raw_docs

    def hybrid_retrieval (self, user_query_processed_list, rewrite_query, do_keyword_search, do_semantic_search, do_RRF, do_cross_encoder, top_i_keyword_search = 64, top_j_semantic_search = 24):

        if not do_keyword_search and not do_semantic_search:
            return []

        multi_query_keyword_chunks = []
        multi_query_semantic_chunks = []

        for user_query_processed in user_query_processed_list:
            if do_keyword_search:
                keyword_chunks = self.inverted_index.search (user_query_processed, top_i_keyword_search)
                multi_query_keyword_chunks.append (keyword_chunks)
            
            if do_semantic_search:
                semantic_chunks = self.vector_database.search (user_query_processed, top_j_semantic_search)
                multi_query_semantic_chunks.append (semantic_chunks)
        
        if do_keyword_search and do_RRF:
            final_keyword_chunks = self.merge_multi_query_retrieval (multi_query_keyword_chunks)
        else:
            final_keyword_chunks = [chunk for results in multi_query_keyword_chunks for chunk in results]

        if do_semantic_search and do_RRF:
            final_semantic_chunks = self.merge_multi_query_retrieval (multi_query_semantic_chunks)
        else:
            final_semantic_chunks = [chunk for results in multi_query_semantic_chunks for chunk in results]

        if do_cross_encoder:
            final_top_k_chunks = self.merge_hybrid_query_retrieval (rewrite_query, final_keyword_chunks, final_semantic_chunks)
        else:
            i, j = 0, 0
            final_top_k_chunks = []

            while i < len (final_keyword_chunks) and j < len (final_semantic_chunks):
                final_top_k_chunks.append (final_keyword_chunks[i]) 
                final_top_k_chunks.append (final_semantic_chunks[j])
                i += 1
                j += 1
            
            while i < len (final_keyword_chunks):
                final_top_k_chunks.append (final_keyword_chunks[i])
                i += 1
            
            while j < len (final_semantic_chunks):
                final_top_k_chunks.append (final_semantic_chunks[j])
                j += 1

        return final_top_k_chunks
        
    def linearize_entity_relationship (self, array_of_relationship, max_relationship = 3):

        disease_precaution_text = []
        disease_medication_text = []
        disease_description_text = []

        for json in array_of_relationship:

            entity1 = json["entity1"]
            entity1_type = json["entity1_type"]
            connection = json["connection"]
            entity2 = json["entity2"]
            entity2_type = json["entity2_type"]

            if entity1 == None or entity1_type == None or connection == None or entity2 == None or entity2_type == None:
                continue

            text = entity1 + f" (type: {entity1_type}) " + connection + " " + entity2 + f" (type: {entity2_type}). " 
            
            if connection == "alert":
                if len (disease_precaution_text) < max_relationship:
                    disease_precaution_text.append (text)

            elif connection == "treated_with":
                if len (disease_medication_text) < max_relationship:
                    disease_medication_text.append (text)

            elif connection == "has_context_of":
                if len (disease_description_text) < max_relationship:
                    disease_description_text.append (text)

        merge_linearized_list = disease_precaution_text + disease_medication_text + disease_description_text
        return merge_linearized_list

    def merge_multi_subgraph_cross_encoder (self, multi_query_graph_chunks, rewritten_query, keep_top_k_chunk = 5):

        if not multi_query_graph_chunks:
            return ""

        pairs = []
        for chunk in multi_query_graph_chunks:
            pairs.append ([chunk, rewritten_query])

        cls_scores = self.rerank_model.predict (pairs)
        pairs = zip (multi_query_graph_chunks, cls_scores)

        key = lambda pair: pair[1]
        zip_list_sort = sorted (pairs, key = key, reverse = True)

        sub_graph_text = ""
        keep_top_k_chunk = min (keep_top_k_chunk, len (zip_list_sort))

        for index in range (keep_top_k_chunk):
            pair = zip_list_sort[index]
            text = pair[0]
            sub_graph_text += text

        return sub_graph_text

    def graph_retrieve (self, entities_list, rewritten_query, do_graph_search, do_cross_encoder):

        if not do_graph_search:
            return ""

        multi_query_graph_chunks = []
        for entities in entities_list:

            array_of_relationship = self.knowledge_database.search (entities)
            if len (array_of_relationship) != 0:
                sub_graph_linearized_list = self.linearize_entity_relationship (array_of_relationship)
                multi_query_graph_chunks.extend (sub_graph_linearized_list)
        
        if len (multi_query_graph_chunks) > 10:
            multi_query_graph_chunks = multi_query_graph_chunks[ : 10]
        if do_cross_encoder:
            multi_query_graph_chunks = self.merge_multi_subgraph_cross_encoder (multi_query_graph_chunks, rewritten_query)

        return multi_query_graph_chunks
