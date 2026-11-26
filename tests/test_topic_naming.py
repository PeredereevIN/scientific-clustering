import unittest

import pandas as pd

from sciintel.topic_naming import (
    build_topic_name,
    detect_ontology_concepts,
    select_naming_terms,
    translate_term,
    translation_source,
)


class TopicNamingTests(unittest.TestCase):
    def test_known_scientific_terms_are_translated_to_russian(self):
        self.assertEqual(translate_term("retrieval augmented generation"),
                         "генерация с дополнением извлечённой информацией (RAG)")
        self.assertEqual(translate_term("vision-language models"), "визуально-языковые модели")
        self.assertEqual(translate_term("motion planning"), "планирование движений")

    def test_unknown_terms_are_preserved_instead_of_word_by_word_mistranslation(self):
        self.assertEqual(translate_term("visual reasoning methods"), "Visual Reasoning Methods")

    def test_aliases_are_canonicalized_before_naming(self):
        frame = pd.DataFrame([
            {"phrase": "robotic manipulation", "canonical_phrase": "robot manipulation",
             "support_docs": 18, "prevalence": .30, "title_support_docs": 8,
             "title_prevalence": .13, "contrast_score": .9, "score": .8, "ctfidf": .7},
            {"phrase": "world model", "canonical_phrase": "world model",
             "support_docs": 12, "prevalence": .20, "title_support_docs": 5,
             "title_prevalence": .08, "contrast_score": .7, "score": .7, "ctfidf": .6},
        ])
        selected = select_naming_terms(frame, 60)
        self.assertEqual(selected[0]["canonical_phrase"], "robot manipulation")
        profile = build_topic_name(frame, 60, "MODERATE_PROFILE")
        self.assertIn("Робототехника", profile["automatic_label_ru"])

    def test_reporting_boilerplate_cannot_become_public_topic_name(self):
        frame = pd.DataFrame([
            {"phrase": "united states", "canonical_phrase": "united states",
             "support_docs": 20, "prevalence": .25, "title_support_docs": 0,
             "title_prevalence": 0, "contrast_score": 1, "score": .9, "ctfidf": .9},
            {"phrase": "artificial intelligence economics", "canonical_phrase": "artificial intelligence economics",
             "support_docs": 9, "prevalence": .12, "title_support_docs": 4,
             "title_prevalence": .06, "contrast_score": .9, "score": .7, "ctfidf": .7},
        ])
        selected = select_naming_terms(frame, 80)
        self.assertNotIn("united states", [x["canonical_phrase"] for x in selected])
        self.assertIn("artificial intelligence economics", [x["canonical_phrase"] for x in selected])

    def test_ai_and_the_economy_is_detected_despite_stop_words(self):
        concepts = detect_ontology_concepts(
            "AI and the Economy: An Economic Examination of Production, Firms and Labor", ""
        )
        self.assertIn("artificial intelligence economics", concepts)

    def test_recommender_terms_form_a_natural_russian_title(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "recommender systems", "support_docs": 30, "prevalence": .3,
             "title_support_docs": 8, "title_prevalence": .08, "contrast_score": 1, "score": .9, "ctfidf": .8},
            {"canonical_phrase": "generative recommendation", "support_docs": 20, "prevalence": .2,
             "title_support_docs": 12, "title_prevalence": .12, "contrast_score": 1, "score": .85, "ctfidf": .7},
            {"canonical_phrase": "semantic ids", "support_docs": 14, "prevalence": .14,
             "title_support_docs": 6, "title_prevalence": .06, "contrast_score": 1, "score": .75, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 100, "MODERATE_PROFILE")["automatic_label_ru"]
        self.assertEqual(label, "Генеративные рекомендательные системы и семантические идентификаторы")


    def test_robotics_world_model_terms_form_natural_russian_title(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "robotic manipulation", "support_docs": 18, "prevalence": .075,
             "title_support_docs": 4, "title_prevalence": .017, "contrast_score": .84, "score": .75, "ctfidf": .8},
            {"canonical_phrase": "world action", "support_docs": 11, "prevalence": .046,
             "title_support_docs": 6, "title_prevalence": .025, "contrast_score": .67, "score": .70, "ctfidf": .7},
            {"canonical_phrase": "world model", "support_docs": 16, "prevalence": .067,
             "title_support_docs": 5, "title_prevalence": .021, "contrast_score": .49, "score": .65, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 239, "MODERATE_PROFILE")["automatic_label_ru"]
        self.assertEqual(label, "Робототехника и модели взаимодействия с окружающим миром")

    def test_agent_terms_form_natural_russian_title(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "llm agents", "support_docs": 16, "prevalence": .083,
             "title_support_docs": 6, "title_prevalence": .031, "contrast_score": .48, "score": .70, "ctfidf": .7},
            {"canonical_phrase": "coding agents", "support_docs": 8, "prevalence": .042,
             "title_support_docs": 4, "title_prevalence": .021, "contrast_score": .45, "score": .60, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 192, "MODERATE_PROFILE")["automatic_label_ru"]
        self.assertEqual(label, "Интеллектуальные агенты и автономные системы")

    def test_language_model_architecture_terms_are_distinguished_from_rag(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "language models", "support_docs": 29, "prevalence": .238,
             "title_support_docs": 15, "title_prevalence": .123, "contrast_score": .0, "score": .88, "ctfidf": .8},
            {"canonical_phrase": "neural networks", "support_docs": 20, "prevalence": .164,
             "title_support_docs": 7, "title_prevalence": .057, "contrast_score": .64, "score": .75, "ctfidf": .7},
            {"canonical_phrase": "speculative decoding", "support_docs": 6, "prevalence": .049,
             "title_support_docs": 6, "title_prevalence": .049, "contrast_score": .51, "score": .70, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 122, "MODERATE_PROFILE")["automatic_label_ru"]
        self.assertEqual(label, "Архитектуры языковых моделей и нейросетевые методы")

    def test_abstains_when_only_non_topical_phrases_exist(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "percentage points", "support_docs": 18, "prevalence": .3,
             "title_support_docs": 0, "title_prevalence": 0, "contrast_score": .9, "score": .9, "ctfidf": .9},
            {"canonical_phrase": "united states", "support_docs": 12, "prevalence": .2,
             "title_support_docs": 0, "title_prevalence": 0, "contrast_score": .8, "score": .8, "ctfidf": .8},
        ])
        label = build_topic_name(frame, 60, "MIXED_TOPIC_REVIEW")["automatic_label_ru"]
        self.assertIn("тематическая проверка", label)


    def test_name_confidence_is_independent_of_cluster_profile_status(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "language models", "support_docs": 85, "prevalence": .333,
             "title_support_docs": 25, "title_prevalence": .098, "contrast_score": .154,
             "score": .79, "ctfidf": .8},
            {"canonical_phrase": "retrieval augmented generation", "support_docs": 18, "prevalence": .071,
             "title_support_docs": 3, "title_prevalence": .012, "contrast_score": .58,
             "score": .44, "ctfidf": .7},
            {"canonical_phrase": "question answering", "support_docs": 17, "prevalence": .067,
             "title_support_docs": 8, "title_prevalence": .031, "contrast_score": .10,
             "score": .33, "ctfidf": .6},
        ])
        review = build_topic_name(frame, 255, "REVIEW_PROFILE")
        moderate = build_topic_name(frame, 255, "MODERATE_PROFILE")
        self.assertEqual(review["naming_confidence"], "Высокая")
        self.assertEqual(review["naming_confidence"], moderate["naming_confidence"])
        self.assertEqual(review["naming_confidence_score"], moderate["naming_confidence_score"])
        self.assertNotEqual(review["top_term_naming_score"], review["naming_confidence_score"])
        self.assertIn("не измеряет однородность", review["naming_confidence_basis"])

    def test_weak_video_terms_do_not_overstate_vision_language_label(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "vision-language models", "support_docs": 25, "prevalence": .184,
             "title_support_docs": 8, "title_prevalence": .059, "contrast_score": .679,
             "score": .70, "ctfidf": .8},
            {"canonical_phrase": "video generation", "support_docs": 7, "prevalence": .051,
             "title_support_docs": 1, "title_prevalence": .007, "contrast_score": .598,
             "score": .45, "ctfidf": .7},
            {"canonical_phrase": "visual quality", "support_docs": 6, "prevalence": .044,
             "title_support_docs": 0, "title_prevalence": 0, "contrast_score": .541,
             "score": .41, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 136)["automatic_label_ru"]
        self.assertEqual(label, "Визуально-языковые модели")

    def test_weak_reinforcement_learning_does_not_dominate_optimization_label(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "strongly convex", "support_docs": 14, "prevalence": .045,
             "title_support_docs": 1, "title_prevalence": .003, "contrast_score": .666,
             "score": .46, "ctfidf": .8},
            {"canonical_phrase": "optimal control", "support_docs": 13, "prevalence": .042,
             "title_support_docs": 4, "title_prevalence": .013, "contrast_score": .311,
             "score": .34, "ctfidf": .7},
            {"canonical_phrase": "reinforcement learning", "support_docs": 16, "prevalence": .051,
             "title_support_docs": 11, "title_prevalence": .035, "contrast_score": 0,
             "score": .28, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 312)["automatic_label_ru"]
        self.assertEqual(label, "Методы оптимизации и оптимального управления")
        self.assertEqual(translate_term("strongly convex"), "сильная выпуклость")

    def test_isac_and_economic_terms_use_ru_dictionary(self):
        self.assertEqual(
            translate_term("integrated sensing and communications"),
            "интегрированные системы связи и зондирования (ISAC)",
        )
        self.assertEqual(translate_term("generative recommendation"),
                         "генеративные рекомендательные системы")
        self.assertEqual(translate_term("markets"), "рынки")
        self.assertEqual(translation_source("markets"), "терминологический словарь")
        self.assertEqual(translation_source("novel unseen term"), "исходный английский термин")


    def test_vla_and_robotics_terms_form_specific_title(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "vision-language-action models", "support_docs": 42, "prevalence": .18,
             "title_support_docs": 22, "title_prevalence": .09, "contrast_score": .8,
             "score": .8, "ctfidf": .8},
            {"canonical_phrase": "robot manipulation", "support_docs": 17, "prevalence": .071,
             "title_support_docs": 4, "title_prevalence": .017, "contrast_score": .84,
             "score": .6, "ctfidf": .7},
        ])
        label = build_topic_name(frame, 239)["automatic_label_ru"]
        self.assertEqual(label, "Робототехника и визуально-языковые модели действий (VLA)")

    def test_economic_topics_are_composed_from_supported_terms(self):
        frame = pd.DataFrame([
            {"canonical_phrase": "artificial intelligence economics", "support_docs": 15, "prevalence": .092,
             "title_support_docs": 8, "title_prevalence": .049, "contrast_score": .403,
             "score": .6, "ctfidf": .8},
            {"canonical_phrase": "trade", "support_docs": 9, "prevalence": .055,
             "title_support_docs": 6, "title_prevalence": .037, "contrast_score": .596,
             "score": .5, "ctfidf": .7},
            {"canonical_phrase": "labor", "support_docs": 8, "prevalence": .049,
             "title_support_docs": 5, "title_prevalence": .031, "contrast_score": .535,
             "score": .45, "ctfidf": .6},
        ])
        label = build_topic_name(frame, 163)["automatic_label_ru"]
        self.assertEqual(label, "Экономика, рынки и влияние искусственного интеллекта")


if __name__ == "__main__":
    unittest.main()
