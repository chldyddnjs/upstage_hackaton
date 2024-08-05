from typing import List, Dict, Any
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
import networkx as nx
from llama_index.core.graph_stores import SimplePropertyGraphStore
from graspologic.partition import hierarchical_leiden
from llama_index.core.query_engine import CustomQueryEngine

class GraphRAGStore(SimplePropertyGraphStore):
    def __init__(self, llm, max_cluster_size=5):
        super().__init__()
        self.llm = llm
        self.max_cluster_size = max_cluster_size
        self.community_summaries = {}

    def build_graph_and_communities(self):
        nx_graph = self._create_networkx_graph()
        communities = self._detect_communities(nx_graph)
        self._generate_community_summaries(nx_graph, communities)

    def _create_networkx_graph(self):
        G = nx.Graph()
        for node in self.graph.nodes.values():
            G.add_node(node.name, **node.properties)
        for relation in self.graph.relations.values():
            G.add_edge(relation.source_id, relation.target_id, **relation.properties)
        return G

    def _detect_communities(self, G):
        return hierarchical_leiden(G, max_cluster_size=self.max_cluster_size)

    def _generate_community_summaries(self, G, communities):
        for community in communities:
            nodes = [n for n in community if n in G.nodes]
            subgraph = G.subgraph(nodes)
            summary = self._summarize_community(subgraph)
            self.community_summaries[frozenset(nodes)] = summary

    def _summarize_community(self, subgraph):
        node_info = "\n".join([f"{n}: {G.nodes[n]}" for n in subgraph.nodes])
        edge_info = "\n".join([f"{u} -> {v}: {G.edges[u, v]}" for u, v in subgraph.edges])
        prompt = f"Summarize the following community:\nNodes:\n{node_info}\nRelationships:\n{edge_info}"
        return self.llm.complete(prompt)

    def get_community_summaries(self):
        if not self.community_summaries:
            self.build_graph_and_communities()
        return self.community_summaries

class GraphRAGExtractor(TransformComponent):
    def __init__(self, llm, max_entities=10, max_relations=5):
        self.llm = llm
        self.max_entities = max_entities
        self.max_relations = max_relations

    def extract_entities_and_relations(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        prompt = f"""
        Analyze the following text and extract:
        1. Up to {self.max_entities} entities with their types and descriptions.
        2. Up to {self.max_relations} relationships between these entities.

        Text: {text}

        Format your response as JSON:
        {{
            "entities": [
                {{"name": "Entity1", "type": "Type1", "description": "Description1"}},
                ...
            ],
            "relations": [
                {{"source": "Entity1", "target": "Entity2", "relation": "RelationType", "description": "RelationDescription"}},
                ...
            ]
        }}
        """
        response = self.llm.complete(prompt)
        return eval(response)  # Convert string to dictionary

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            text = node.get_content()
            extracted_data = self.extract_entities_and_relations(text)
            
            entities = [
                EntityNode(name=e['name'], label=e['type'], properties={"description": e['description']})
                for e in extracted_data['entities']
            ]
            relations = [
                Relation(source_id=r['source'], target_id=r['target'], label=r['relation'], properties={"description": r['description']})
                for r in extracted_data['relations']
            ]
            
            node.metadata[KG_NODES_KEY] = entities
            node.metadata[KG_RELATIONS_KEY] = relations
        
        return nodes

class GraphRAGQueryEngine(CustomQueryEngine):
    def __init__(self, graph_store: GraphRAGStore, llm):
        self.graph_store = graph_store
        self.llm = llm

    def custom_query(self, query_str: str) -> str:
        summaries = self.graph_store.get_community_summaries()
        community_answers = [self._query_community(summary, query_str) for summary in summaries.values()]
        return self._aggregate_answers(community_answers, query_str)

    def _query_community(self, community_summary: str, query: str) -> str:
        prompt = f"""
        Given the following community summary:
        {community_summary}

        Answer this query: {query}

        Provide a concise answer based solely on the information in the community summary.
        """
        return self.llm.complete(prompt)

    def _aggregate_answers(self, community_answers: List[str], original_query: str) -> str:
        combined_answers = "\n".join(community_answers)
        prompt = f"""
        Original query: {original_query}

        Community-specific answers:
        {combined_answers}

        Provide a comprehensive and coherent answer to the original query by synthesizing the information from all community answers.
        """
        return self.llm.complete(prompt)

