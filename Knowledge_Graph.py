from neo4j import GraphDatabase
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import joblib
import os
import re


class Knowledge_Graphbase:

    def __init__ (self, connection_URI = None, user = "neo4j"):

        load_dotenv ()
        password = os.getenv ("AUTH")
        connection_URI = connection_URI or os.getenv ("NEO4J_URI") or "bolt://my-neo4j:7687"
        self.driver = GraphDatabase.driver (connection_URI, auth = (user, password))

    def knowledge_graph_workflow_1 (self, writer, disease_name, medication, precaution):

        for pre in precaution:
            writer.run ("""
                    MERGE (p: Precaution {name: $p_name})
                    MERGE (d: Disease {name: $d_name})
                    MERGE (p)-[:alert]->(d)
                """, p_name = pre.lower ().strip (), d_name = disease_name.lower ().strip ())   

        for med in medication:
            writer.run ("""
                    MERGE (d: Disease {name: $d_name})
                    MERGE (m: Medication {name: $m_name}) 
                    MERGE (d)-[:treated_with]->(m)
                """, d_name = disease_name.lower ().strip (), m_name = med.lower ().strip ())

    def knowledge_graph_workflow_2 (self, writer, parent_chunk, idx):
        
        for index in range (len (parent_chunk)):
            child_chunk = parent_chunk[index]
            text_content = child_chunk.page_content
            metadata = child_chunk.metadata 
            disease_name = metadata["disease_name"]
            source = metadata["source"]

            writer.run ("""
                    MERGE (d: Disease {name: $d_name})
                    MERGE (c: Chunk {name: $text, parent_chunk_id: $id1, child_chunk_id: $id2, source: $source})
                    MERGE (d)-[:has_context_of]->(c)
                """, d_name = disease_name.lower ().strip (), text = text_content.lower ().strip (), id1 = idx, id2 = index, source = source.lower ().strip ()) 

    def construct_knowledge_graph (self, dataset, chunks):

        with self.driver.session () as session:
            
            for row in dataset.toLocalIterator ():
                disease_name = row["json_dataset"]['disease_name']
                medication = row["json_dataset"]["disease_treatment_plan"]
                precaution = row["json_dataset"]["disease_precautions"]

                medication = re.findall (r'[A-Z][a-zA-Z0-9\-\s]+', medication)
                medication = [string.strip () for string in medication]

                precaution = re.findall (r'[a-zA-Z0-9][a-zA-Z0-9\-\s]+', precaution)
                precaution = [string.strip () for string in precaution]    

                session.execute_write (self.knowledge_graph_workflow_1, disease_name, medication, precaution)

            for idx in range (len (chunks)):
                parent_chunk = chunks[idx]
                session.execute_write (self.knowledge_graph_workflow_2, parent_chunk, idx)     

    def retrieval_graphbase (self, retriever, entities):

        query = """
                    MATCH (n)
                    WHERE (n: Disease AND n.name IN $disease_entities)
                    OR (n: Medication AND n.name IN $medication_entities)
                    
                    OPTIONAL MATCH (n)-[edge]-(child)
                    
                    RETURN n.name AS entity1,
                            labels(n)[0] AS entity1_type,
                            type(edge) AS connection,
                            child.name AS entity2,
                            child.parent_chunk_id AS parent_chunk_id,
                            child.child_chunk_id AS child_chunk_id,
                            child.source AS source,
                            labels(child)[0] AS entity2_type
                """
        array_of_relationship = retriever.run (query, disease_entities = entities[0], medication_entities = entities[1])

        return array_of_relationship.data ()

    def search (self, entities):

        with self.driver.session () as session:
            array_of_relationship = session.execute_read (self.retrieval_graphbase, entities)

        return array_of_relationship

    def save_local (self, path = "neo4j.cypher"):

        with self.driver.session () as session:
            result = session.run ("""
                                CALL apoc.export.cypher.all (null, 
                                {
                                stream: true, 
                                format: "cypher-shell", 
                                useTypes: true
                                })
                            """)
            
            with open (path, "w") as file:
                for relationship in result:
                    file.write (relationship["cypherStatements"])

    def load_local (self, path = "neo4j.cypher"):

        if not os.path.isfile (path):
            raise FileNotFoundError (f"Neo4j snapshot not found: {path}")
        with open (path, "rb") as snapshot:
            if snapshot.read (64).startswith (b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError (f"Neo4j snapshot is a Git LFS pointer: {path}")

        with self.driver.session() as session:
            procedure = session.run (
                "SHOW PROCEDURES YIELD name WHERE name = 'apoc.cypher.runFile' RETURN count(*) AS available"
            ).single ()
            if not procedure or procedure["available"] == 0:
                raise RuntimeError ("APOC Extended procedure apoc.cypher.runFile is unavailable")

            session.run ("MATCH (n) DETACH DELETE n")
            session.run ("CALL apoc.schema.assert ({}, {})")
            session.run ("CALL apoc.cypher.runFile ($path)", path = os.path.basename (path)).consume ()


if __name__ == '__main__':

    spark = SparkSession.builder.appName ('Parquet').getOrCreate ()

    dataset = spark.read.parquet ("Processed_Dataset.parquet")
    chunks = joblib.load ("Chunks.pkl")

    knowledge_graph = Knowledge_Graphbase ()
    knowledge_graph.construct_knowledge_graph (dataset, chunks)
    knowledge_graph.save_local ()
