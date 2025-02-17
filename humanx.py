from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from datetime import datetime, timedelta
import re
import os
import time
import json
import torch
import requests
import subprocess
import smtplib
import shutil
import asyncio
import warnings
import logging
import dateparser
import psutil  # To check system memory
from email.mime.text import MIMEText
import nest_asyncio

# For hierarchical clustering
from scipy.cluster.hierarchy import linkage, fcluster

# Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator

# Suppress warnings and only log errors
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)
torch.set_num_threads(16)

# ====================================================
# AGI Features: MetaCognitive System (Learning & Reasoning)
# ====================================================

class ThoughtLevel(Enum):
    REFLEXIVE = 1    # Immediate responses
    ANALYTICAL = 2   # Logical analysis
    METACOGNITIVE = 3  # Thinking about thinking
    CREATIVE = 4     # Novel connections
    ABSTRACT = 5     # High-level concepts

@dataclass
class MetaCognitiveBelief:
    concept: str
    confidence: float
    evidence: List[str]
    contradictions: List[str]
    last_updated: datetime
    abstraction_level: int

class MetaCognitiveSystem:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.belief_system: Dict[str, MetaCognitiveBelief] = {}
        self.learning_rate = 0.1
        self.abstraction_threshold = 0.7
        self.vectorizer = TfidfVectorizer()
        self.meta_memory = defaultdict(list)
        self.reasoning_patterns: List[Dict[str, Any]] = []
        self.performance_metrics = defaultdict(list)
        # Storage for all conversation data (for self-training)
        self.training_data: List[str] = []

    def metacognitive_process(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Phase 1: Self-reflection
        current_state = self._assess_current_state(input_data)
        # Phase 2: Problem decomposition
        subproblems = self._decompose_problem(input_data)
        # Phase 3: Abstract reasoning
        abstract_concepts = self._generate_abstractions(subproblems)
        # Phase 4: Symbolic manipulation and reasoning
        symbolic_results = self._symbolic_reasoning(abstract_concepts)
        # Phase 5: Self-improvement
        self._update_reasoning_patterns(symbolic_results)
        return self._synthesize_results(current_state, symbolic_results)

    def _assess_current_state(self, input_data: str) -> Dict[str, float]:
        metrics = {
            'reasoning_efficiency': self._calculate_reasoning_efficiency(),
            'learning_progress': self._evaluate_learning_progress(),
            'abstraction_capability': self._measure_abstraction_capability(input_data),
            'belief_consistency': self._check_belief_consistency()
        }
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)
        return metrics


    def _symbolic_reasoning(self, abstractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Performs symbolic reasoning on the given abstractions."""
        results = []
        for abstraction in abstractions:
            symbols = self._create_symbols(abstraction)
            transformed = self._apply_transformations(symbols)
            consistent = self._verify_consistency(transformed)
            insights = self._generate_insights(transformed)
            confidence = self._calculate_confidence(insights)
            results.append({
                'original': abstraction,
                'symbols': symbols,
                'transformed': transformed,
                'consistent': consistent,
                'insights': insights,
                'confidence': confidence
            })
        return results

    def _calculate_reasoning_efficiency(self) -> float:
        if self.reasoning_patterns:
            total_conf = sum(p.get('effectiveness', 0) for p in self.reasoning_patterns)
            return min(1.0, total_conf / len(self.reasoning_patterns))
        return 0.5

    def _evaluate_learning_progress(self) -> float:
        metrics = self.performance_metrics.get('abstraction_capability', [])
        if len(metrics) >= 2:
            return max(0.0, metrics[-1] - metrics[-2])
        return 0.1

    def _measure_abstraction_capability(self, input_data: str) -> float:
        try:
            vec = self.vectorizer.fit_transform([input_data])
            variance = np.var(vec.toarray())
            return min(1.0, variance / 0.5)
        except Exception:
            return 0.5

    def _check_belief_consistency(self) -> float:
        if not self.belief_system:
            return 1.0
        total, count = 0.0, 0
        for belief in self.belief_system.values():
            total += 0.5 if belief.contradictions else 1.0
            count += 1
        return total / count if count else 1.0

    # ----- Problem Decomposition -----
    def _decompose_problem(self, problem: str) -> List[Dict[str, Any]]:
        components = self._extract_problem_components(problem)
        dependency_graph = self._build_dependency_graph(components)
        subproblems = []
        for comp in nx.topological_sort(dependency_graph):
            subproblems.append({
                'component': comp,
                'dependencies': list(dependency_graph.predecessors(comp)),
                'complexity': self._estimate_complexity(comp),
                'abstraction_level': self._determine_abstraction_level(comp)
            })
        return subproblems

    def _extract_problem_components(self, problem: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.?!]', problem) if s.strip()]

    def _build_dependency_graph(self, components: List[str]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for comp in components:
            graph.add_node(comp)
        # For simplicity, no dependency edges are added.
        return graph

    def _estimate_complexity(self, component: str) -> float:
        return min(1.0, len(component) / 100.0)

    def _determine_abstraction_level(self, component: str) -> int:
        wc = len(component.split())
        if wc > 15:
            return ThoughtLevel.ABSTRACT.value
        elif wc > 10:
            return ThoughtLevel.CREATIVE.value
        elif wc > 5:
            return ThoughtLevel.METACOGNITIVE.value
        return ThoughtLevel.REFLEXIVE.value

    # ----- Abstraction Generation -----
    def _vectorize_elements(self, concrete_elements: List[Dict[str, Any]]) -> np.ndarray:
        texts = [elem['component'] for elem in concrete_elements]
        features = self.vectorizer.fit_transform(texts)
        return features.toarray()

    def _hierarchical_clustering(self, features: np.ndarray) -> List[Dict[str, Any]]:
        if len(features) == 0:
            return []
        Z = linkage(features, method='ward')
        labels = fcluster(Z, t=2, criterion='maxclust')
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(i)
        result = []
        for label, indices in clusters.items():
            # In a real system, level could be computed from features.
            result.append({'elements': indices, 'level': np.mean(indices)})
        return result

    def _extract_commonalities(self, elements: List[Dict[str, Any]]) -> str:
        if not elements:
            return ""
        common = set(elements[0]['component'].split())
        for elem in elements[1:]:
            common &= set(elem['component'].split())
        return " ".join(common)

    def _identify_relations(self, elements: List[Dict[str, Any]]) -> str:
        return "interrelated"

    def _generate_abstractions(self, concrete_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        features = self._vectorize_elements(concrete_elements)
        if len(features) < 2:
            clusters = [{'elements': concrete_elements, 'level': np.mean([elem['abstraction_level'] for elem in concrete_elements])}]
        else:
            raw_clusters = self._hierarchical_clustering(features)
            clusters = [{
                'elements': [concrete_elements[i] for i in cluster['elements']],
                'level': np.mean([concrete_elements[i]['abstraction_level'] for i in cluster['elements']])
            } for cluster in raw_clusters]
        abstractions = []
        for cluster in clusters:
            abstraction = {
                'concepts': cluster['elements'],
                'level': cluster['level'],
                'commonalities': self._extract_commonalities(cluster['elements']),
                'relations': self._identify_relations(cluster['elements'])
            }
            abstractions.append(abstraction)
            self._update_knowledge_graph(abstraction)
        return abstractions

    def _update_knowledge_graph(self, abstraction: Dict[str, Any]):
        self.knowledge_graph.add_node(str(abstraction))

    # ----- Symbolic Reasoning -----
    def _create_symbols(self, abstraction: Dict[str, Any]) -> Dict[str, Any]:
        symbols = {}
        for elem in abstraction['concepts']:
            comp = elem['component']
            symbols[comp] = "".join(word[0].upper() for word in comp.split() if word)
        return symbols

    def _apply_transformations(self, symbols: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v + "_XFORM" for k, v in symbols.items()}

    def _verify_consistency(self, transformed: Dict[str, Any]) -> bool:
        return all(bool(val) for val in transformed.values())

    def _generate_insights(self, transformed: Dict[str, Any]) -> List[str]:
        return [f"Insight: {k} becomes {v}" for k, v in transformed.items()]

    def _calculate_confidence(self, insights: List[str]) -> float:
        if not insights:
            return 0.0
        avg_len = sum(len(s) for s in insights) / len(insights)
        return min(1.0, avg_len / 100.0)

    # ----- Reasoning Pattern Updates -----
    def _update_reasoning_patterns(self, results: List[Dict[str, Any]]):
        for result in results:
            if result['confidence'] > self.abstraction_threshold:
                pattern = {
                    'type': self._identify_pattern_type(result),
                    'preconditions': self._extract_preconditions(result),
                    'transformations': result['transformed'],
                    'effectiveness': result['confidence']
                }
                self.reasoning_patterns.append(pattern)
        self._prune_patterns()
        self._generate_meta_patterns()

    def _identify_pattern_type(self, result: Dict[str, Any]) -> str:
        return "strong" if result['confidence'] > 0.8 else "moderate"

    def _extract_preconditions(self, result: Dict[str, Any]) -> str:
        texts = [elem['component'] for elem in result['original']['concepts']]
        common = set(texts[0].split())
        for t in texts[1:]:
            common &= set(t.split())
        return " ".join(common)

    def _prune_patterns(self):
        self.reasoning_patterns = [p for p in self.reasoning_patterns if p['effectiveness'] > 0.5]

    def _generate_meta_patterns(self):
        if len(self.reasoning_patterns) > 1:
            merged_type = "meta_" + "_".join(p['type'] for p in self.reasoning_patterns)
            avg_eff = sum(p['effectiveness'] for p in self.reasoning_patterns) / len(self.reasoning_patterns)
            self.reasoning_patterns.append({
                'type': merged_type,
                'preconditions': "combined",
                'transformations': {},
                'effectiveness': min(1.0, avg_eff)
            })

    def improve_reasoning(self) -> None:
        trends = self._analyze_performance_trends()
        improvement_areas = self._identify_improvement_areas(trends)
        self._adjust_parameters(improvement_areas)
        self._generate_new_strategies(improvement_areas)

    def _analyze_performance_trends(self) -> Dict[str, float]:
        trends = {}
        for metric, values in self.performance_metrics.items():
            if len(values) > 1:
                x = np.arange(len(values))
                y = np.array(values)
                trend = np.polyfit(x, y, 1)[0]
                trends[metric] = trend
        return trends

    def _identify_improvement_areas(self, trends: Dict[str, float]) -> List[str]:
        return [k for k, t in trends.items() if t < 0]

    def _adjust_parameters(self, improvement_areas: List[str]) -> None:
        if improvement_areas:
            self.learning_rate = max(0.01, self.learning_rate * 0.95)

    def _create_pattern_variations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        variations = []
        for pattern in patterns:
            new_pat = pattern.copy()
            new_pat['effectiveness'] = min(1.0, pattern['effectiveness'] + 0.05)
            variations.append(new_pat)
        return variations

    def _validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        is_consistent = bool(pattern.get('preconditions'))
        is_complete = bool(pattern.get('type')) and bool(pattern.get('transformations'))
        return is_consistent and is_complete and pattern.get('effectiveness', 0) > 0.7

    def _generate_new_strategies(self, improvement_areas: List[str]) -> None:
        for area in improvement_areas:
            successful = [p for p in self.reasoning_patterns if p['effectiveness'] > 0.8]
            new_patterns = self._create_pattern_variations(successful)
            for pat in new_patterns:
                if self._validate_pattern(pat):
                    self.reasoning_patterns.append(pat)

    def _synthesize_results(self, state: Dict[str, float], symbolic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        synthesis = {
            'cognitive_state': state,
            'abstract_insights': self._extract_key_insights(symbolic_results),
            'confidence': self._aggregate_confidence(symbolic_results),
            'novel_patterns': self._identify_novel_patterns(symbolic_results),
            'recommendations': self._generate_recommendations(state, symbolic_results)
        }
        self._update_beliefs(synthesis)
        return synthesis

    def _extract_key_insights(self, symbolic_results: List[Dict[str, Any]]) -> List[str]:
        insights = []
        for result in symbolic_results:
            insights.extend(result.get('insights', []))
        return insights

    def _aggregate_confidence(self, symbolic_results: List[Dict[str, Any]]) -> float:
        if symbolic_results:
            return sum(r.get('confidence', 0) for r in symbolic_results) / len(symbolic_results)
        return 0.0

    def _identify_novel_patterns(self, symbolic_results: List[Dict[str, Any]]) -> List[str]:
        return [f"Pattern_{i}" for i in range(len(symbolic_results))]

    def _generate_recommendations(self, state: Dict[str, float], symbolic_results: List[Dict[str, Any]]) -> List[str]:
        recs = []
        if state['reasoning_efficiency'] < 0.7:
            recs.append("Reexamine the problem structure to improve reasoning efficiency.")
        if state['learning_progress'] < 0.2:
            recs.append("Review past conversation errors to boost learning progress.")
        if not symbolic_results:
            recs.append("No insights were generated; consider rephrasing the query.")
        return recs

    def _update_beliefs(self, synthesis: Dict[str, Any]):
        self.belief_system['last_synthesis'] = MetaCognitiveBelief(
            concept="synthesis",
            confidence=synthesis['confidence'],
            evidence=synthesis['abstract_insights'],
            contradictions=[],
            last_updated=datetime.now(),
            abstraction_level=int(synthesis['cognitive_state']['abstraction_capability'] * 5)
        )

    # ----- New Self-Training Method for Complete Autonomous Learning -----
    def self_train(self, new_data: List[str]) -> None:
        """
        Incorporate new conversation data to update internal knowledge.
        This method re-fits the TF-IDF vectorizer on all accumulated training data and
        updates internal reasoning patterns.
        """
        # Store new conversation inputs
        self.training_data.extend(new_data)
        try:
            # Update the vectorizer with the full training data
            self.vectorizer.fit(self.training_data)
            # Optionally, one could also re-run clustering over all meta_memory or training_data
            # to update the knowledge graph and reasoning patterns.
            print("Self-training complete. Updated internal knowledge.")
        except Exception as e:
            print("Error during self-training:", e)

# ====================================================
# AGI Features: Cognitive Architecture (Memory & Reasoning)
# ====================================================

@dataclass
class Memory:
    content: str
    timestamp: datetime
    importance: float
    associations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

class CognitiveArchitecture:
    def __init__(self):
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: Dict[str, Memory] = {}
        self.memory_threshold = 0.5
        self.max_stm_size = 10
        self.reasoning_confidence = 0.0

    def process_input(self, input_text: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        complexity = self._analyze_complexity(input_text)
        emotion = self._analyze_emotional_content(input_text)
        mem = Memory(
            content=input_text,
            timestamp=datetime.now(),
            importance=complexity,
            context=current_context
        )
        self._manage_memory(mem)
        reasoning = self._logical_reasoning(input_text, self.short_term_memory)
        return {
            'complexity': complexity,
            'emotional_content': emotion,
            'reasoning_result': reasoning,
            'memory_context': self._get_relevant_memories(input_text)
        }

    def _analyze_complexity(self, text: str) -> float:
        words = text.split()
        avg_len = sum(len(w) for w in words) / len(words) if words else 0
        sentences = text.split('.')
        avg_sent = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        return min(1.0, (avg_len / 10 + avg_sent / 20) / 2)

    def _analyze_emotional_content(self, text: str) -> Dict[str, float]:
        emotions = {
            'joy': ['happy', 'glad', 'joyful', 'delighted'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'depressed'],
            'anger': ['angry', 'furious', 'annoyed', 'irritated'],
            'fear': ['afraid', 'scared', 'worried', 'anxious']
        }
        text_lower = text.lower()
        scores = {}
        for emo, keywords in emotions.items():
            score = sum(text_lower.count(k) for k in keywords)
            scores[emo] = min(1.0, score / len(text.split()))
        return scores

    def _logical_reasoning(self, input_text: str, memories: List[Memory]) -> Dict[str, Any]:
        concepts = self._extract_concepts(input_text)
        patterns = self._identify_patterns(concepts, memories)
        conclusions = self._generate_conclusions(patterns)
        return {
            'concepts': concepts,
            'patterns': patterns,
            'conclusions': conclusions,
            'confidence': self.reasoning_confidence
        }

    def _extract_concepts(self, text: str) -> List[str]:
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return list(set(w for w in words if w not in stopwords and len(w) > 3))

    def _identify_patterns(self, concepts: List[str], memories: List[Memory]) -> List[Dict[str, Any]]:
        patterns = []
        mem_concepts = []
        for mem in memories:
            mem_concepts.extend(self._extract_concepts(mem.content))
        freq = {}
        for c in mem_concepts:
            freq[c] = freq.get(c, 0) + 1
        for c in concepts:
            if c in freq and freq[c] > 1:
                patterns.append({
                    'concept': c,
                    'frequency': freq[c],
                    'significance': freq[c] / len(memories)
                })
        return patterns

    def _generate_conclusions(self, patterns: List[Dict[str, Any]]) -> List[str]:
        conclusions = []
        total_sig = sum(p['significance'] for p in patterns)
        if patterns:
            self.reasoning_confidence = min(1.0, total_sig / len(patterns))
            for p in patterns:
                if p['significance'] > 0.5:
                    conclusions.append(f"Strong pattern: {p['concept']}")
                elif p['significance'] > 0.3:
                    conclusions.append(f"Moderate pattern: {p['concept']}")
        return conclusions

    def _manage_memory(self, memory: Memory):
        self.short_term_memory.append(memory)
        if len(self.short_term_memory) > self.max_stm_size:
            oldest = self.short_term_memory.pop(0)
            if oldest.importance > self.memory_threshold:
                self.long_term_memory[str(oldest.timestamp)] = oldest

    def _get_relevant_memories(self, input_text: str) -> List[Memory]:
        inp_concepts = set(self._extract_concepts(input_text))
        relevant = []
        all_mem = self.short_term_memory + list(self.long_term_memory.values())
        for mem in all_mem:
            mem_concepts = set(self._extract_concepts(mem.content))
            rel = len(inp_concepts.intersection(mem_concepts)) / len(inp_concepts) if inp_concepts else 0
            if rel > 0.3:
                relevant.append(mem)
        return relevant

    def consolidate_memories(self):
        now = datetime.now()
        new_stm = []
        for mem in self.short_term_memory:
            if (now - mem.timestamp).total_seconds() > 60:
                self.long_term_memory[str(mem.timestamp)] = mem
            else:
                new_stm.append(mem)
        self.short_term_memory = new_stm

# ====================================================
# Chatbot with GPT‑Neo and Integrated AGI (No Operator Features)
# ====================================================

@dataclass
class ChatbotConfig:
    model_name: str = "EleutherAI/gpt-neo-1.3B"
    max_new_tokens: int = 100      # For GPU
    cpu_max_new_tokens: int = 50   # For CPU
    temperature: float = 0.2

class Chatbot:
    def __init__(self):
        print("HumanX is loading. Please wait...")
        print("HumanX is free and open-source. This AI runs best on a TPU or GPU.")
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.config = ChatbotConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        if self.device.type == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = self.accelerator.prepare(model)
        self.model.eval()
        # Simple sentiment analyzer to gauge user tone
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        # Check if system has at least 12GB of RAM
        if psutil.virtual_memory().total >= 16 * 1024**3:
            self.agi_enabled = True
            print("HumanX AGI enabled.")
        else:
            self.device = torch.device("cpu")
            model.to(self.device) # Move model to CPU
            self.agi_enabled = False
            print("HumanX AGI disabled. GPU required.")
        # Initialize cognitive modules
        self.meta_cognitive_system = MetaCognitiveSystem()
        self.cognitive_architecture = CognitiveArchitecture()
        self.interaction_count = 0
        print("HumanX is ready.")

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        att_mask = inputs.attention_mask.to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=att_mask,
                max_new_tokens=50 if self.device.type == "cpu" else 100,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.5
            )
        return self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    def learn_from_conversation(self, user_input: str):
        # Use sentiment analysis to gauge negative feedback and adjust internal beliefs.
        sentiment = self.sentiment_analyzer(user_input)[0]
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.6:
            last_belief = self.meta_cognitive_system.belief_system.get('last_synthesis')
            if last_belief:
                last_belief.confidence *= 0.9
        # Trigger self-improvement
        self.meta_cognitive_system.improve_reasoning()
        # Also, let the system completely learn by self-training on new data.
        self.meta_cognitive_system.self_train([user_input])

    async def chat(self, user_input: str) -> str:
        if self.agi_enabled:
            cognitive = self.cognitive_architecture.process_input(user_input, {})
            meta = self.meta_cognitive_system.metacognitive_process(user_input, cognitive)
            combined_context = (
                f"Cognitive Analysis: {json.dumps(cognitive)}\n"
                f"MetaCognitive Insights: {json.dumps(meta)}\n"
                f"User said: {user_input}"
            )
            prompt = f"User: {user_input}\nContext: {combined_context}\nChatbot:\n"
            response = self.generate_response(prompt)
        else:
            response = self.generate_response(f"User: {user_input}\nChatbot:\n")
        self.learn_from_conversation(user_input)
        self.interaction_count += 1
        if self.interaction_count % 5 == 0:
            self.cognitive_architecture.consolidate_memories()
        return response

    async def run(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye!")
                break
            response = await self.chat(user_input)
            print(f"Chatbot: {response}")

async def main():
    bot = Chatbot()
    await bot.run()

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Allow nested event loops
    asyncio.run(main())


