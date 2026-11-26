from __future__ import annotations

import logging

from . import db
from .config import settings

log = logging.getLogger("sciintel")


def export_neo4j(run_id: str | None = None, batch_size: int = 500) -> dict:
    if not settings.neo4j_password:
        raise RuntimeError("Set NEO4J_PASSWORD in .env before exporting")
    from neo4j import GraphDatabase

    papers = db.rows("SELECT id, title, domain, path, doi, year FROM documents")
    terms = db.rows("""
      SELECT document_id, phrase, method, score, rank FROM keywords
    """)
    labels = []
    similarity_edges = []
    if run_id:
        labels = db.rows("SELECT document_id, cluster_id FROM cluster_labels WHERE run_id=?", (run_id,))
        edge_path = settings.artifacts / "runs" / run_id / "graph_edges.csv"
        if edge_path.exists():
            import pandas as pd
            similarity_edges = pd.read_csv(edge_path).to_dict("records")
    with GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)) as driver:
        driver.verify_connectivity()
        driver.execute_query(
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            database_=settings.neo4j_database)
        driver.execute_query(
            "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
            database_=settings.neo4j_database)
        for start in range(0, len(papers), batch_size):
            driver.execute_query("""
              UNWIND $rows AS row
              MERGE (p:Paper {id: row.id})
              SET p.title=row.title, p.domain=row.domain, p.path=row.path,
                  p.doi=row.doi, p.year=row.year
            """, parameters_={"rows": papers[start:start + batch_size]}, database_=settings.neo4j_database)
        for start in range(0, len(terms), batch_size):
            driver.execute_query("""
              UNWIND $rows AS row
              MATCH (p:Paper {id: row.document_id})
              MERGE (k:Keyword {name: toLower(row.phrase)})
              SET k.label=row.phrase
              MERGE (p)-[r:HAS_KEYWORD {method: row.method}]->(k)
              SET r.score=row.score, r.rank=row.rank
            """, parameters_={"rows": terms[start:start + batch_size]}, database_=settings.neo4j_database)
        for start in range(0, len(labels), batch_size):
            driver.execute_query("""
              UNWIND $rows AS row
              MATCH (p:Paper {id: row.document_id})
              MERGE (c:Cluster {run_id: $run_id, cluster_id: row.cluster_id})
              MERGE (p)-[:ASSIGNED_TO]->(c)
            """, parameters_={"rows": labels[start:start + batch_size], "run_id": run_id},
                                   database_=settings.neo4j_database)
        for start in range(0, len(similarity_edges), batch_size):
            driver.execute_query("""
              UNWIND $rows AS row
              MATCH (a:Paper {id: row.source_id})
              MATCH (b:Paper {id: row.target_id})
              MERGE (a)-[r:SIMILAR_TO]->(b)
              SET r.cosine_similarity = row.similarity, r.run_id = $run_id
            """, parameters_={"rows": similarity_edges[start:start + batch_size], "run_id": run_id},
                                   database_=settings.neo4j_database)
    result = {"papers": len(papers), "keyword_edges": len(terms), "cluster_assignments": len(labels),
              "similarity_edges": len(similarity_edges)}
    log.info("Neo4j export complete: %s", result)
    return result
