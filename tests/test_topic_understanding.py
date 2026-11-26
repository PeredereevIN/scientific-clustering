import unittest
import pandas as pd

from sciintel.topic_understanding import ClusterTopicUnderstanding, TopicConfig, BOUNDARY


class TopicUnderstandingTests(unittest.TestCase):
    def test_extracts_explicit_abstract_and_stops_before_introduction(self):
        engine = ClusterTopicUnderstanding()
        raw = """Paper title\nJane Doe\nDepartment of Computer Science\nAbstract\nWe study speech recognition and acoustic modeling for multilingual audio.\nOur method improves word error rate.\n1 Introduction\nThe introduction contains unrelated graph algorithms and treewidth."""
        abstract, mode = engine._extract_abstract(raw, "Paper title")
        self.assertEqual(mode, "explicit_abstract")
        self.assertIn("speech recognition", abstract.lower())
        self.assertNotIn("treewidth", abstract.lower())

    def test_aliases_canonicalize_to_one_topic(self):
        self.assertEqual(ClusterTopicUnderstanding._canonical("ASR"), "speech recognition")
        self.assertEqual(ClusterTopicUnderstanding._canonical("VLMs"), "vision-language models")
        self.assertEqual(ClusterTopicUnderstanding._canonical("GRPO"), "group relative policy optimization")

    def test_phrase_extraction_does_not_cross_title_abstract_boundary(self):
        engine = ClusterTopicUnderstanding()
        phrases = list(engine._segments(f"speech recognition {BOUNDARY} acoustic models"))
        self.assertIn("speech recognition", phrases)
        self.assertIn("acoustic models", phrases)
        self.assertNotIn(f"recognition {BOUNDARY} acoustic", phrases)

    def test_ignores_noise_assignments(self):
        assignments = pd.DataFrame([
            {"document_id": "1", "group_id": "A", "title": "Speech Recognition", "domain": "nlp"},
            {"document_id": "2", "group_id": "A", "title": "Acoustic Speech Recognition", "domain": "nlp"},
            {"document_id": "3", "group_id": "NOISE", "title": "Unrelated graph paper", "domain": "ml"},
        ])
        texts = pd.DataFrame([
            {"document_id": "1", "text": "Abstract\nWe study speech recognition and acoustic models for audio.\n1 Introduction\nGraph algorithms."},
            {"document_id": "2", "text": "Abstract\nSpeech recognition models use acoustic features and audio signals.\n1 Introduction\nGraph theory."},
            {"document_id": "3", "text": "Abstract\nA completely different topic.\n1 Introduction"},
        ])
        engine = ClusterTopicUnderstanding(TopicConfig(min_primary_prevalence=0.1, min_secondary_prevalence=0.1))
        result = engine.fit_predict(assignments, texts)
        self.assertEqual(result["profiles"]["group_id"].tolist(), ["A"])
        noise = result["documents"].set_index("document_id").loc["3"]
        self.assertEqual(noise["automatic_label"], "Не распределено")

    def test_topic_profile_is_derived_from_document_content_not_cluster_id(self):
        rows = []
        texts = []
        for i in range(8):
            rows.append({"document_id": f"s{i}", "group_id": "arbitrary_7", "title": f"Speech Recognition with Acoustic Models {i}"})
            texts.append({"document_id": f"s{i}", "text": "Abstract\nSpeech recognition uses acoustic models for audio signals and word error rate evaluation.\n1 Introduction\nOther material."})
        for i in range(8):
            rows.append({"document_id": f"g{i}", "group_id": "arbitrary_2", "title": f"Graph Algorithms and Shortest Paths {i}"})
            texts.append({"document_id": f"g{i}", "text": "Abstract\nGraph algorithms study shortest paths, graph traversal and vertex connectivity.\n1 Introduction\nOther material."})
        engine = ClusterTopicUnderstanding(TopicConfig(min_primary_prevalence=0.1, min_secondary_prevalence=0.08, min_subtopic_group_size=100))
        result = engine.fit_predict(pd.DataFrame(rows), pd.DataFrame(texts))
        profiles = result["profiles"].set_index("group_id")
        self.assertIn("speech recognition", profiles.loc["arbitrary_7", "automatic_label_en"].lower())
        self.assertIn("graph", profiles.loc["arbitrary_2", "automatic_label_en"].lower())
        self.assertNotEqual(profiles.loc["arbitrary_7", "automatic_label_en"], profiles.loc["arbitrary_2", "automatic_label_en"])

    def test_no_abstract_uses_flagged_intro_fallback(self):
        engine = ClusterTopicUnderstanding()
        abstract, mode = engine._extract_abstract("Title\nThis paper studies graph algorithms and shortest paths." * 20, "Title")
        self.assertIn(mode, {"intro_fallback", "title_only"})

    def test_bibliography_is_not_in_abstract(self):
        engine = ClusterTopicUnderstanding()
        raw = "Abstract\nWe investigate retrieval augmented generation and knowledge graphs.\nReferences\nSmith et al. Graph retrieval systems."
        abstract, mode = engine._extract_abstract(raw, "Some title")
        self.assertNotIn("Smith", abstract)

    def test_detects_internal_subtopics_without_changing_parent_group(self):
        rows = []
        texts = []
        for i in range(18):
            doc_id = f"speech_{i}"
            rows.append({"document_id": doc_id, "group_id": "one_parent_cluster", "title": f"Speech Recognition and Acoustic Models {i}"})
            texts.append({"document_id": doc_id, "text": "Abstract\nSpeech recognition uses acoustic models, audio signals, speech synthesis and word error rate.\n1 Introduction\nMore."})
        for i in range(18):
            doc_id = f"graph_{i}"
            rows.append({"document_id": doc_id, "group_id": "one_parent_cluster", "title": f"Graph Algorithms and Shortest Paths {i}"})
            texts.append({"document_id": doc_id, "text": "Abstract\nGraph algorithms study shortest paths, graph traversal, lower bounds and approximation algorithms.\n1 Introduction\nMore."})
        engine = ClusterTopicUnderstanding(TopicConfig(min_subtopic_group_size=24, min_subtopic_silhouette=0.10))
        result = engine.fit_predict(pd.DataFrame(rows), pd.DataFrame(texts))
        profile = result["profiles"].iloc[0]
        self.assertEqual(profile["group_id"], "one_parent_cluster")
        self.assertEqual(profile["topic_profile_status"], "INTERNAL_SUBTOPICS_DETECTED")
        self.assertIn("Несколько тематических направлений", profile["automatic_label"])
        self.assertIn("Mixed research directions", profile["automatic_label_en"])
        self.assertEqual(len(result["subtopics"]), 2)
        self.assertEqual(result["subtopic_assignments"]["group_id"].nunique(), 1)

    def test_profile_exports_russian_name_and_evidence_without_changing_group_ids(self):
        rows = []
        texts = []
        variations = ["wages", "employment", "productivity", "pricing", "firms",
                      "workforce", "production", "welfare", "adoption", "distribution"]
        for i, variation in enumerate(variations):
            rows.append({"document_id": f"econ{i}", "group_id": "community_5",
                         "title": f"AI and the Economy: Labor Market {variation}"})
            texts.append({"document_id": f"econ{i}", "text": (
                f"Abstract\nWe examine artificial intelligence adoption, economic productivity, "
                f"labor markets, firm behavior and algorithmic pricing in the context of {variation}.\n"
                "1 Introduction\nA longer introduction about unrelated methods is not used."
            )})
        result = ClusterTopicUnderstanding().fit_predict(pd.DataFrame(rows), pd.DataFrame(texts))
        profile = result["profiles"].iloc[0]
        self.assertEqual(profile["group_id"], "community_5")
        self.assertIn("экономика", profile["automatic_label_ru"].lower())
        self.assertTrue(profile["automatic_label_en"])
        self.assertTrue(profile["naming_evidence"])
        self.assertEqual(set(result["documents"]["group_id"]), {"community_5"})


    def test_abstract_only_ontology_evidence_does_not_inflate_title_support(self):
        rows = []
        texts = []
        for i in range(6):
            rows.append({"document_id": f"econ_abs_{i}", "group_id": "economics", "title": f"A Study of Production {i}"})
            texts.append({"document_id": f"econ_abs_{i}", "text": (
                "Abstract\nWe examine AI and the economy, productivity, labor markets, "
                "firm behavior and welfare.\n1 Introduction\nNot used."
            )})
        for i in range(6):
            rows.append({"document_id": f"graph_abs_{i}", "group_id": "graphs", "title": f"Graph Structures {i}"})
            texts.append({"document_id": f"graph_abs_{i}", "text": (
                "Abstract\nWe study graph traversal, shortest paths and vertex connectivity.\n1 Introduction\nNot used."
            )})
        result = ClusterTopicUnderstanding().fit_predict(pd.DataFrame(rows), pd.DataFrame(texts))
        candidates = result["candidates"]
        concept = candidates[(candidates["group_id"] == "economics") &
                             (candidates["canonical_phrase"] == "artificial intelligence economics")]
        self.assertFalse(concept.empty)
        self.assertEqual(int(concept.iloc[0]["support_docs"]), 6)
        self.assertEqual(int(concept.iloc[0]["title_support_docs"]), 0)
        self.assertEqual(int(concept.iloc[0]["literal_support_docs"]), 0)
        self.assertEqual(int(concept.iloc[0]["ontology_only_support_docs"]), 6)

    def test_title_ontology_evidence_is_counted_as_title_support(self):
        rows = []
        texts = []
        for i in range(6):
            rows.append({"document_id": f"econ_title_{i}", "group_id": "economics", "title": f"AI and the Economy: Production {i}"})
            texts.append({"document_id": f"econ_title_{i}", "text": (
                "Abstract\nWe examine productivity, labor markets, firm behavior and welfare.\n1 Introduction\nNot used."
            )})
        for i in range(6):
            rows.append({"document_id": f"graph_title_{i}", "group_id": "graphs", "title": f"Graph Structures {i}"})
            texts.append({"document_id": f"graph_title_{i}", "text": (
                "Abstract\nWe study graph traversal, shortest paths and vertex connectivity.\n1 Introduction\nNot used."
            )})
        result = ClusterTopicUnderstanding().fit_predict(pd.DataFrame(rows), pd.DataFrame(texts))
        candidates = result["candidates"]
        concept = candidates[(candidates["group_id"] == "economics") &
                             (candidates["canonical_phrase"] == "artificial intelligence economics")]
        self.assertFalse(concept.empty)
        self.assertEqual(int(concept.iloc[0]["title_support_docs"]), 6)
        self.assertEqual(int(concept.iloc[0]["literal_title_support_docs"]), 0)
        self.assertEqual(int(concept.iloc[0]["ontology_title_support_docs"]), 6)

    def test_topic_phrases_require_contiguous_multiword_concepts(self):
        self.assertTrue(ClusterTopicUnderstanding._valid_phrase("speech recognition"))
        self.assertTrue(ClusterTopicUnderstanding._valid_phrase("knowledge graphs"))
        self.assertFalse(ClusterTopicUnderstanding._valid_phrase("retrieval"))
        self.assertFalse(ClusterTopicUnderstanding._valid_phrase("information processing systems neurips"))
        self.assertFalse(ClusterTopicUnderstanding._valid_phrase("set number"))


    def test_economic_single_word_concepts_can_be_candidates(self):
        for term in ("market", "markets", "trade", "labor", "labour", "inequality"):
            self.assertTrue(ClusterTopicUnderstanding._valid_phrase(term), term)

    def test_vla_and_isac_are_detected_as_normalized_concepts(self):
        from sciintel.topic_naming import detect_ontology_concepts
        vla = detect_ontology_concepts(
            "Vision-Language-Action Models for Long-Horizon Robot Manipulation", ""
        )
        isac = detect_ontology_concepts(
            "Joint Power Optimization for Integrated Sensing and Communications (ISAC)", ""
        )
        self.assertIn("vision-language-action models", vla)
        self.assertIn("integrated sensing and communications", isac)


if __name__ == "__main__":
    unittest.main()
