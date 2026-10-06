from Hybrid_Dual_Indexing import Keyword_Search, Semantic_Search
from Knowledge_Graph import Knowledge_Graphbase
from Vector_Database import Vector_DB
from sentence_transformers import CrossEncoder
import torch
import joblib
import json
import os
import logging
from itertools import zip_longest


class Retriever:

    @staticmethod
    def document_key (doc):
        metadata = getattr (doc, "metadata", {}) or {}
        return (doc.page_content, json.dumps (metadata, sort_keys = True, default = str))

    @staticmethod
    def load_semantic_model (path = "Semantic_Model.pkl"):
        original_storage_init = torch.UntypedStorage.__new__
        had_mps_deserialize = hasattr (torch.serialization, '_mps_deserialize')

        def patched_storage_new (cls, *args, **kwargs):
            if kwargs.get ('device') == 'mps':
                kwargs['device'] = 'cpu'
            return original_storage_init (cls, *args, **kwargs)

        torch.UntypedStorage.__new__ = patched_storage_new
        if not had_mps_deserialize:
            torch.serialization._mps_deserialize = lambda obj, location: obj.cpu ()

        try:
            try:
                model = joblib.load (path)
            except Exception:
                with open (path, "rb") as file:
                    model = torch.load (file, map_location = 'cpu', weights_only = False)

            if hasattr (model, 'to'):
                model.to ('cpu')
            return model
        finally:
            torch.UntypedStorage.__new__ = original_storage_init
            if not had_mps_deserialize:
                delattr (torch.serialization, '_mps_deserialize')

    def __init__ (self, restore_graph_snapshot = None):

        self.startup_warnings = []
        self.semantic_search_model = None
        self.vector_database = None
        self.inverted_index = None
        self.rerank_model = None
        self.knowledge_database = None

        try:
            self.semantic_search_model = self.load_semantic_model ()
            self.vector_database = Vector_DB (self.semantic_search_model, "LOAD_DATABASE")
        except Exception as error:
            self.startup_warnings.append ("Semantic search is unavailable; the FAISS path could not load.")
            logging.getLogger (__name__).warning ("Semantic startup failed: %s", type (error).__name__)

        try:
            self.inverted_index = joblib.load ("Keyword_Model.pkl")
        except Exception as error:
            self.startup_warnings.append ("Keyword search is unavailable; the BM25 index could not load.")
            logging.getLogger (__name__).warning ("Keyword startup failed: %s", type (error).__name__)

        try:
            self.rerank_model = CrossEncoder ("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as error:
            self.startup_warnings.append ("Reranking is unavailable; results use retrieval order.")
            logging.getLogger (__name__).warning ("Reranker startup failed: %s", type (error).__name__)

        if restore_graph_snapshot is None:
            restore_graph_snapshot = os.getenv ("RESTORE_GRAPH_SNAPSHOT", "false").lower () in ("1", "true", "yes", "on")
        try:
            self.knowledge_database = Knowledge_Graphbase ()
            if restore_graph_snapshot:
                self.knowledge_database.load_local ()
        except Exception as error:
            if restore_graph_snapshot:
                raise
            self.startup_warnings.append ("Graph search is unavailable; Neo4j could not initialize.")
            logging.getLogger (__name__).warning ("Graph startup failed: %s", type (error).__name__)

    def merge_multi_query_retrieval (self, multi_query_retrieval, decay_rank = 60, keep_top_k_chunk = 10):

        rank_docs = {}
        raw_docs_mapping = {}

        for each_query_retrieval in multi_query_retrieval:
            for rank, doc in enumerate (each_query_retrieval):

                doc_rank = 1 / (decay_rank + rank)
                doc_key = self.document_key (doc)
                raw_docs_mapping[doc_key] = doc

                if doc_key in rank_docs:
                    rank_docs[doc_key] += doc_rank
                else:
                    rank_docs[doc_key] = doc_rank

        key = lambda item: item[1]
        rank_docs = sorted (rank_docs.items (), key = key, reverse = True)

        raw_docs = []
        keep_top_k_chunk = min (keep_top_k_chunk, len (rank_docs))

        for index in range (keep_top_k_chunk):

            doc_key = rank_docs[index][0]
            raw_doc = raw_docs_mapping[doc_key]
            raw_docs.append (raw_doc)
        
        return raw_docs

    def merge_hybrid_query_retrieval (self, rewrite_query, keyword_chunks, semantic_chunks, keep_top_k_chunk = 5):

        pairs = []
        merge_docs = []

        seen = set ()
        for chunk in (keyword_chunks or []) + (semantic_chunks or []):
            doc_key = self.document_key (chunk)
            if doc_key in seen:
                continue
            seen.add (doc_key)
            pairs.append ([rewrite_query, chunk.page_content])
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

    def interleave_unique (self, keyword_chunks, semantic_chunks):

        merged = []
        seen = set ()
        for pair in zip_longest (keyword_chunks, semantic_chunks):
            for chunk in pair:
                if chunk is None:
                    continue
                key = self.document_key (chunk)
                if key not in seen:
                    seen.add (key)
                    merged.append (chunk)
        return merged

    def hybrid_retrieval (self, user_query_processed_list, rewrite_query, do_keyword_search, do_semantic_search, do_RRF, do_cross_encoder, top_i_keyword_search = 64, top_j_semantic_search = 24, return_warnings = False):

        warnings = getattr (self, "startup_warnings", []).copy ()

        if do_keyword_search and self.inverted_index is None:
            do_keyword_search = False
        if do_semantic_search and self.vector_database is None:
            do_semantic_search = False
        if do_cross_encoder and self.rerank_model is None:
            do_cross_encoder = False

        if not do_keyword_search and not do_semantic_search:
            return ([], warnings) if return_warnings else []

        multi_query_keyword_chunks = []
        multi_query_semantic_chunks = []

        for user_query_processed in user_query_processed_list:
            if do_keyword_search:
                try:
                    keyword_chunks = self.inverted_index.search (user_query_processed, top_i_keyword_search)
                    multi_query_keyword_chunks.append (keyword_chunks)
                except Exception as error:
                    if "Some keyword searches failed; available results were used." not in warnings:
                        warnings.append ("Some keyword searches failed; available results were used.")
                    logging.getLogger (__name__).warning ("Keyword search failed: %s", type (error).__name__)
            
            if do_semantic_search:
                try:
                    semantic_chunks = self.vector_database.search (user_query_processed, top_j_semantic_search)
                    multi_query_semantic_chunks.append (semantic_chunks)
                except Exception as error:
                    if "Some semantic searches failed; available results were used." not in warnings:
                        warnings.append ("Some semantic searches failed; available results were used.")
                    logging.getLogger (__name__).warning ("Semantic search failed: %s", type (error).__name__)
        
        if do_keyword_search and do_RRF:
            final_keyword_chunks = self.merge_multi_query_retrieval (multi_query_keyword_chunks)
        else:
            final_keyword_chunks = [chunk for results in multi_query_keyword_chunks for chunk in results]

        if do_semantic_search and do_RRF:
            final_semantic_chunks = self.merge_multi_query_retrieval (multi_query_semantic_chunks)
        else:
            final_semantic_chunks = [chunk for results in multi_query_semantic_chunks for chunk in results]

        if do_cross_encoder:
            try:
                final_top_k_chunks = self.merge_hybrid_query_retrieval (rewrite_query, final_keyword_chunks, final_semantic_chunks)
            except Exception as error:
                warnings.append ("Reranking was unavailable; results use retrieval order.")
                logging.getLogger (__name__).warning ("Reranking failed: %s", type (error).__name__)
                final_top_k_chunks = self.interleave_unique (final_keyword_chunks, final_semantic_chunks)
        else:
            final_top_k_chunks = self.interleave_unique (final_keyword_chunks, final_semantic_chunks)

        final_top_k_chunks = final_top_k_chunks[:5]
        return (final_top_k_chunks, warnings) if return_warnings else final_top_k_chunks
        
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
            pairs.append ([rewritten_query, chunk])

        cls_scores = self.rerank_model.predict (pairs)
        pairs = zip (multi_query_graph_chunks, cls_scores)

        key = lambda pair: pair[1]
        zip_list_sort = sorted (pairs, key = key, reverse = True)

        keep_top_k_chunk = min (keep_top_k_chunk, len (zip_list_sort))
        return "\n".join (pair[0] for pair in zip_list_sort[:keep_top_k_chunk])

    def graph_retrieve (self, entities_list, rewritten_query, do_graph_search, do_cross_encoder):

        if not do_graph_search:
            return ""

        multi_query_graph_chunks = []
        for entities in entities_list:

            array_of_relationship = self.knowledge_database.search (entities)
            if len (array_of_relationship) != 0:
                sub_graph_linearized_list = self.linearize_entity_relationship (array_of_relationship)
                multi_query_graph_chunks.extend (sub_graph_linearized_list)
        
        multi_query_graph_chunks = list (dict.fromkeys (multi_query_graph_chunks))[:10]
        if do_cross_encoder and self.rerank_model is not None:
            try:
                return self.merge_multi_subgraph_cross_encoder (multi_query_graph_chunks, rewritten_query)
            except Exception as error:
                logging.getLogger (__name__).warning ("Graph reranking failed: %s", type (error).__name__)

        return "\n".join (multi_query_graph_chunks[:5])
