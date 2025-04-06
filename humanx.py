# update 2:30 AM its late outside and i wanna sleep
# update 2:40 AM i slept
import os
import re
import json
import time
import torch
import asyncio
import warnings
import logging
import subprocess
import sys
import threading
import concurrent.futures
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from sympy import sympify, SympifyError
import psutil
from flask import Flask, render_template_string, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
from groq import Groq
from github import Github


try:
    import black
except ImportError:
    black = None

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
torch.set_num_threads(16)

# make this idiot rerember thing
chat_history_path = "chat_history.json"

def load_chat_history():
    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as f:
            data = json.load(f)
            # chat sessions
            if isinstance(data, list):
                return data
    # make a chat for the idiot
    return [{
        "id": str(int(time.time())),
        "name": "New Chat",
        "messages": []
    }]

def save_chat_history(history):
    with open(chat_history_path, "w") as f:
        json.dump(history, f)

chat_history = load_chat_history()


class ThoughtLevel(Enum):
    REFLEXIVE = 1
    ANALYTICAL = 2
    METACOGNITIVE = 3
    CREATIVE = 4
    ABSTRACT = 5

@dataclass
class belief:
    concept: str
    confidence: float
    evidence: list
    contradictions: list
    last_updated: datetime
    abstraction_level: int



class brain:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.belief_system: dict = {}
        self.learning_rate = 0.1
        self.abstraction_threshold = 0.7
        self.vectorizer = TfidfVectorizer()
        self.meta_memory = defaultdict(list)
        self.reasoning_patterns: list = []
        self.performance_metrics = defaultdict(list)
        self.training_data: list = []

    def metacognitive_process(self, input_data: str, context: dict) -> dict:
        current_state = self._assess_current_state(input_data)
        subproblems = self._decompose_problem(input_data)
        abstract_concepts = self._generate_abstractions(subproblems)
        symbolic_results = self._symbolic_reasoning(abstract_concepts)
        self._update_reasoning_patterns(symbolic_results)
        return self._synthesize_results(current_state, symbolic_results)

    def _assess_current_state(self, input_data: str) -> dict:
        metrics = {
            'reasoning_efficiency': self._calculate_reasoning_efficiency(),
            'learning_progress': self._evaluate_learning_progress(),
            'abstraction_capability': self._measure_abstraction_capability(input_data),
            'belief_consistency': self._check_belief_consistency()
        }
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)
        return metrics

    def _symbolic_reasoning(self, abstractions: list) -> list:
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

    def _decompose_problem(self, problem: str) -> list:
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

    def _extract_problem_components(self, problem: str) -> list:
        return [s.strip() for s in re.split(r'[.?!]', problem) if s.strip()]

    def _build_dependency_graph(self, components: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        for comp in components:
            graph.add_node(comp)
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

    def _vectorize_elements(self, concrete_elements: list) -> np.ndarray:
        texts = [elem['component'] for elem in concrete_elements]
        features = self.vectorizer.fit_transform(texts)
        return features.toarray()

    def _hierarchical_clustering(self, features: np.ndarray) -> list:
        if len(features) == 0:
            return []
        Z = linkage(features, method='ward')
        labels = fcluster(Z, t=2, criterion='maxclust')
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(i)
        result = []
        for label, indices in clusters.items():
            result.append({'elements': indices, 'level': np.mean(indices)})
        return result

    def _extract_commonalities(self, elements: list) -> str:
        if not elements:
            return ""
        common = set(elements[0]['component'].split())
        for elem in elements[1:]:
            common &= set(elem['component'].split())
        return " ".join(common)

    def _identify_relations(self, elements: list) -> str:
        return "interrelated"

    def _generate_abstractions(self, concrete_elements: list) -> list:
        features = self._vectorize_elements(concrete_elements)
        if len(features) < 2:
            clusters = [{
                'elements': concrete_elements,
                'level': np.mean([elem['abstraction_level'] for elem in concrete_elements])
            }]
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

    def _update_knowledge_graph(self, abstraction: dict):
        self.knowledge_graph.add_node(str(abstraction))

    def _create_symbols(self, abstraction: dict) -> dict:
        symbols = {}
        for elem in abstraction['concepts']:
            comp = elem['component']
            symbols[comp] = "".join(word[0].upper() for word in comp.split() if word)
        return symbols

    def _apply_transformations(self, symbols: dict) -> dict:
        return {k: v + "_XFORM" for k, v in symbols.items()}

    def _verify_consistency(self, transformed: dict) -> bool:
        return all(bool(val) for val in transformed.values())

    def _generate_insights(self, transformed: dict) -> list:
        return [f"Insight: {k} becomes {v}" for k, v in transformed.items()]

    def _calculate_confidence(self, insights: list) -> float:
        if not insights:
            return 0.0
        avg_len = sum(len(s) for s in insights) / len(insights)
        return min(1.0, avg_len / 100.0)

    def _update_reasoning_patterns(self, results: list):
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

    def _identify_pattern_type(self, result: dict) -> str:
        return "strong" if result['confidence'] > 0.8 else "moderate"

    def _extract_preconditions(self, result: dict) -> str:
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

    def _synthesize_results(self, state: dict, symbolic_results: list) -> dict:
        synthesis = {
            'cognitive_state': state,
            'abstract_insights': [ins for result in symbolic_results for ins in result.get('insights', [])],
            'confidence': (sum(r.get('confidence', 0) for r in symbolic_results) / len(symbolic_results)) if symbolic_results else 0.0,
            'novel_patterns': [f"Pattern_{i}" for i in range(len(symbolic_results))],
            'recommendations': self._generate_recommendations(state, symbolic_results)
        }
        self._update_beliefs(synthesis)
        return synthesis

    def _generate_recommendations(self, state: dict, symbolic_results: list) -> list:
        recs = []
        if state['reasoning_efficiency'] < 0.7:
            recs.append("Reexamine the problem structure to improve reasoning efficiency.")
        if state['learning_progress'] < 0.2:
            recs.append("Review past conversation errors to boost learning progress.")
        if not symbolic_results:
            recs.append("No insights were generated; consider rephrasing the query.")
        return recs

    def _update_beliefs(self, synthesis: dict):
        self.belief_system['last_synthesis'] = belief(
            concept="synthesis",
            confidence=synthesis['confidence'],
            evidence=synthesis['abstract_insights'],
            contradictions=[],
            last_updated=datetime.now(),
            abstraction_level=int(synthesis['cognitive_state']['abstraction_capability'] * 5)
        )

    def _analyze_performance_trends(self) -> dict:
        trends = {}
        for metric, values in self.performance_metrics.items():
            if len(values) > 1:
                x = np.arange(len(values))
                y = np.array(values)
                trend = np.polyfit(x, y, 1)[0]
                trends[metric] = trend
        return trends

    def _identify_improvement_areas(self, trends: dict) -> list:
        return [k for k, t in trends.items() if t < 0]

    def _adjust_parameters(self, improvement_areas: list) -> None:
        if improvement_areas:
            self.learning_rate = max(0.01, self.learning_rate * 0.95)

    def _create_pattern_variations(self, patterns: list) -> list:
        variations = []
        for pattern in patterns:
            new_pat = pattern.copy()
            new_pat['effectiveness'] = min(1.0, pattern['effectiveness'] + 0.05)
            variations.append(new_pat)
        return variations

    def _validate_pattern(self, pattern: dict) -> bool:
        is_consistent = bool(pattern.get('preconditions'))
        is_complete = bool(pattern.get('type')) and bool(pattern.get('transformations'))
        return is_consistent and is_complete and pattern.get('effectiveness', 0) > 0.7

    def _generate_new_strategies(self, improvement_areas: list) -> None:
        for area in improvement_areas:
            successful = [p for p in self.reasoning_patterns if p['effectiveness'] > 0.8]
            new_patterns = self._create_pattern_variations(successful)
            for pat in new_patterns:
                if self._validate_pattern(pat):
                    self.reasoning_patterns.append(pat)

    def improve_reasoning(self) -> None:
        trends = self._analyze_performance_trends()
        improvement_areas = self._identify_improvement_areas(trends)
        self._adjust_parameters(improvement_areas)
        self._generate_new_strategies(improvement_areas)

    def self_train(self, new_data: list) -> None:
        self.training_data.extend(new_data)
        try:
            self.vectorizer.fit(self.training_data)
        except Exception as e:
            logging.error("Error during self-training: %s", e)



@dataclass
class Memory:
    content: str
    timestamp: datetime
    importance: float
    associations: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)

class CognitiveArchitecture:
    def __init__(self):
        self.short_term_memory: list = []
        self.long_term_memory: dict = {}
        self.memory_threshold = 0.5
        self.max_stm_size = 10
        self.max_ltm_size = 100
        self.reasoning_confidence = 0.0

    def _serialize_memory(self, memory: Memory) -> dict:
        return {
            "content": memory.content,
            "timestamp": memory.timestamp.isoformat(),
            "importance": memory.importance,
            "associations": memory.associations,
            "context": memory.context,
            "tags": memory.tags
        }

    def process_input(self, input_text: str, current_context: dict) -> dict:
        complexity = self._analyze_complexity(input_text)
        emotion = self._analyze_emotional_content(input_text)
        tags = self._assign_tags(input_text)
        mem = Memory(
            content=input_text,
            timestamp=datetime.now(),
            importance=complexity,
            context=current_context,
            tags=tags
        )
        self._manage_memory(mem)
        reasoning = self._logical_reasoning(input_text, self.short_term_memory)
        return {
            'complexity': complexity,
            'emotional_content': emotion,
            'reasoning_result': reasoning,
            'memory_context': [self._serialize_memory(m) for m in self._get_relevant_memories(input_text, tags)]
        }

    def _assign_tags(self, input_text: str) -> list:
        tags = []
        if self._is_coding_task(input_text):
            tags.append("coding")
        is_math, _, _ = parse_and_solve_math(input_text)
        if is_math:
            tags.append("math")
        if "debug" in input_text.lower():
            tags.append("debugging")
        if "document" in input_text.lower() or "docstring" in input_text.lower():
            tags.append("documentation")
        technical_keywords = ["data", "algorithm", "system", "compute", "network"]
        if any(kw in input_text.lower() for kw in technical_keywords):
            tags.append("technical")
        return tags

    def _is_coding_task(self, text: str) -> bool:
        coding_keywords = ["code", "program", "function", "class", "module", "library", "algorithm", "data structure"]
        return any(keyword in text.lower() for keyword in coding_keywords)

    def _analyze_complexity(self, text: str) -> float:
        words = text.split()
        avg_len = sum(len(w) for w in words) / len(words) if words else 0
        sentences = text.split('.')
        avg_sent = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        return min(1.0, (avg_len / 10 + avg_sent / 20) / 2)
# HumanX AI Emotion Engine code
    def _analyze_emotional_content(self, text: str) -> dict:
        emotions = {
            'joy': ['happy', 'glad', 'joyful', 'delighted','cheerful'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'down'],
            'anger': ['angry', 'furious', 'annoyed', 'irritated'],
            'fear': ['afraid', 'scared', 'worried', 'anxious']
        }
        text_lower = text.lower()
        scores = {}
        for emo, keywords in emotions.items():
            score = sum(text_lower.count(k) for k in keywords)
            scores[emo] = min(1.0, score / len(text.split())) if text.split() else 0.0
        return scores

    def _logical_reasoning(self, input_text: str, memories: list) -> dict:
        concepts = self._extract_concepts(input_text)
        patterns = self._identify_patterns(concepts, memories)
        conclusions = self._generate_conclusions(patterns)
        return {
            'concepts': concepts,
            'patterns': patterns,
            'conclusions': conclusions,
            'confidence': self.reasoning_confidence
        }

    def _extract_concepts(self, text: str) -> list:
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return list(set(w for w in words if w not in stopwords and len(w) > 3))

    def _identify_patterns(self, concepts: list, memories: list) -> list:
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

    def _generate_conclusions(self, patterns: list) -> list:
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
                if len(self.long_term_memory) > self.max_ltm_size:
                    oldest_key = min(self.long_term_memory, key=lambda k: self.long_term_memory[k].timestamp)
                    del self.long_term_memory[oldest_key]

    def _get_relevant_memories(self, input_text: str, domains: list) -> list:
        inp_concepts = set(self._extract_concepts(input_text))
        relevant = []
        all_mem = self.short_term_memory + list(self.long_term_memory.values())
        for mem in all_mem:
            mem_concepts = set(self._extract_concepts(mem.content))
            concept_rel = len(inp_concepts.intersection(mem_concepts)) / len(inp_concepts) if inp_concepts else 0
            tag_rel = len(set(domains).intersection(set(mem.tags))) / len(domains) if domains else 0
            rel = 0.7 * concept_rel + 0.3 * tag_rel
            if rel > 0.3:
                relevant.append((mem, rel))
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in relevant[:5]]

    def consolidate_memories(self):
        now = datetime.now()
        new_stm = []
        for mem in self.short_term_memory:
            if (now - mem.timestamp).total_seconds() > 60:
                self.long_term_memory[str(mem.timestamp)] = mem
                if len(self.long_term_memory) > self.max_ltm_size:
                    oldest_key = min(self.long_term_memory, key=lambda k: self.long_term_memory[k].timestamp)
                    del self.long_term_memory[oldest_key]
            else:
                new_stm.append(mem)
        self.short_term_memory = new_stm

# the idiot can do math!11!!!!!!111!!111

def parse_and_solve_math(user_input: str, include_proof: bool = False) -> tuple:
    try:
        cleaned_input = user_input.replace(' ', '')
        if not any(op in cleaned_input for op in ['+', '-', '*', '/', '^', '=', '>', '<']):
            return False, "", ""
        try:
            expression = sympify(cleaned_input)
            result = str(expression.evalf())
        except (SympifyError, TypeError, ValueError):
            try:
                if not all(c in '0123456789+-*/.() ' for c in cleaned_input):
                    return False, "", ""
                result = str(eval(cleaned_input, {"__builtins__": {}}, {}))
            except:
                return False, "", ""
        explanation = f"Evaluated {user_input} directly using mathematical operations."
        if include_proof:
            explanation += f" For example, starting with {user_input}, we apply order of operations (PEMDAS) to compute step-by-step."
        return True, result, explanation
    except Exception as e:
        logging.error(f"Math parsing error: {str(e)}")
        return False, "", ""



def animate_debugging(stop_event):
    spinner = ['|', '/', '-', '\\']
    while not stop_event.is_set():
        for char in spinner:
            sys.stdout.write(f'\rDebugging... {char}')
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write('\rDebugging... done\n')
    sys.stdout.flush()



def format_code(code: str, language: str) -> str:
    if language.lower() == "python" and black:
        try:
            formatted_code = black.format_file_contents(code, fast=True, mode=black.Mode())
            return formatted_code
        except Exception as e:
            logging.error("Error formatting code: %s", e)
            return code
    return code



def get_system_status() -> str:
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    status = f"CPU Usage: {cpu_percent}%\nMemory Usage: {memory.percent}% (Total: {memory.total // (1024**2)} MB)"
    return status



@dataclass
class ChatbotConfig:
    inference_engine: str = "groq"
    inference_model: str = "qwen-2.5-32b"
    max_new_tokens: int = 6000
    cpu_max_new_tokens: int = 6000
    temperature: float = 0.2
    enable_advanced_reasoning: bool = True
    interactive_learning: bool = False
    advanced_generation_mode: bool = True
    
    def update_model(self, mode: str):
        if mode == "standard":
            self.inference_model = "llama-3.2-90b-vision-preview"
            self.temperature = 0.2
        elif mode == "reason":
            self.inference_model = "qwen-qwq-32b"
            self.temperature = 0.3
            self.enable_advanced_reasoning = True
        elif mode == "flash":
            self.inference_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            self.temperature = 0.1
            self.max_new_tokens = 2000

class Chatbot:
    def __init__(self):
        self.config = ChatbotConfig()
        logging.info("HumanX v1.0.0")
        logging.info("2025 Krishnaa Rajesh")
        self.accelerator = Accelerator(cpu=True)
        self.adaptive_temperature = self.config.temperature
        self.learning_mode_active = False
        self.interaction_count = 0
        self.debug_cache = {}
        self.debug_success_patterns = set()
        
        # use groq
        if self.config.inference_engine == "groq":
            logging.info("Using Groq Library for inference with model %s.", self.config.inference_model)
            try:
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise Exception("GROQ_API_KEY environment variable not set.")
                self.groq_client = Groq(api_key=api_key)
                logging.info("Groq client initialized successfully.")
            except Exception as e:
                logging.error("Error initializing Groq client: %s", e)
                self.groq_client = None
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            self.model = None
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.inference_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.config.inference_model)
            supported_engines = torch.backends.quantized.supported_engines
            logging.info("Supported quantization engines: %s", supported_engines)
            if 'qnnpack' in supported_engines:
                torch.backends.quantized.engine = 'qnnpack'
                logging.info("Using quantization engine: qnnpack")
            elif 'fbgemm' in supported_engines:
                torch.backends.quantized.engine = 'fbgemm'
                logging.info("Using quantization engine: fbgemm")
            else:
                logging.warning("No supported quantization engine found. Proceeding without quantization.")
            try:
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                self.agi_enabled = True
            except RuntimeError as e:
                logging.error("Quantization failed: %s. Proceeding without quantization.", e)
                self.agi_enabled = False
            self.model = self.accelerator.prepare(model)
            self.model.eval()
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            logging.info("HumanX AGI %s.", "enabled" if self.agi_enabled else "disabled")
        
        self.meta_cognitive_system = brain()
        self.cognitive_architecture = CognitiveArchitecture()
        logging.info("HumanX is ready.")

    def groq_inference(self, prompt: str, max_new_tokens: int, memory_context: str = "") -> str:
        if not self.groq_client:
            logging.error("Groq client not initialized.")
            return "Error: Groq client not initialized."
        try:
            messages = [
                {"role": "system", "content": "You are an advanced AGI system with full reasoning, planning, and memory recall capabilities, as well as being able to effectively demonstate emotion, wants, needs and other things. You are to act as a human and be very convincing as one, having emotional intelligence and try your best to be human, and when confronted act as if you are speaking like one. Use the provided memory context when answering."}
            ]
            if memory_context:
                messages.append({"role": "system", "content": f"Memory Context: {memory_context}"})
            messages.append({"role": "user", "content": prompt})
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.config.inference_model,
                max_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=0.95
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error("Error during Groq inference: %s", e)
            return "Error during Groq inference."

    def generate_response(self, prompt: str, hidden_context: str = "", clean_code: bool = False, is_fix_generation: bool = False) -> str:
        if self.config.advanced_generation_mode:
            advanced_context = "\n[Advanced Reasoning Mode Enabled]\n"
            prompt = advanced_context + prompt
        if self._is_coding_task(prompt) and not is_fix_generation:
            new_max_tokens = self.config.max_new_tokens
        elif is_fix_generation:
            new_max_tokens = 500
        else:
            new_max_tokens = self.config.cpu_max_new_tokens if (self.config.inference_engine != "groq") and (torch.device("cpu").type == "cpu") else self.config.max_new_tokens

        if self.config.inference_engine == "groq":
            response_text = self.groq_inference(prompt, new_max_tokens, memory_context=hidden_context)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            att_mask = inputs.attention_mask.to(self.model.device)
            stop_event = None
            animation_thread = None
            if new_max_tokens == self.config.max_new_tokens:
                stop_event = threading.Event()
                animation_thread = threading.Thread(target=animate_debugging, args=(stop_event,))
                animation_thread.start()
            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=att_mask,
                        max_new_tokens=new_max_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        top_p=0.95,
                        temperature=self.adaptive_temperature
                    )
            except Exception as e:
                logging.error("Error during generation: %s", e)
                if stop_event:
                    stop_event.set()
                    animation_thread.join()
                return "Error generating response."
            if stop_event:
                stop_event.set()
                animation_thread.join()
            response_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if clean_code:
            response_text = re.sub(r'```[\w]*', '', response_text).strip()
        return response_text

    def _is_coding_task(self, user_input: str) -> bool:
        coding_keywords = ["code", "program", "function", "class", "module", "library", "algorithm", "data structure"]
        return any(keyword in user_input.lower() for keyword in coding_keywords)

    def _is_debugging_task(self, user_input: str) -> bool:
        return "debug" in user_input.lower()

    def _is_documentation_task(self, user_input: str) -> bool:
        return "document" in user_input.lower() or "docstring" in user_input.lower()

    def _detect_language_preference(self, user_input: str) -> str:
        languages = {"python": "Python", "c++": "C++", "java": "Java"}
        for key, lang in languages.items():
            if key in user_input.lower():
                return lang
        return "Python"

    def _get_file_extension(self, language: str) -> str:
        extensions = {"Python": ".py", "C++": ".cpp", "Java": ".java"}
        return extensions.get(language, ".py")

    def _generate_filename(self, user_input: str) -> str:
        words = user_input.lower().split()
        for word in words:
            if word in ["function", "class", "program"]:
                idx = words.index(word)
                if idx + 1 < len(words):
                    return words[idx + 1]
        return "generated_code"

    def prompt_for_filename(self, default_name: str, language: str) -> str:
        user_input = input(f"Please enter the desired file name (default: {default_name}{self._get_file_extension(language)}): ")
        return user_input.strip() if user_input.strip() else default_name

    def save_code_to_file(self, code: str, language: str, user_prompt: str) -> str:
        default_name = self._generate_filename(user_prompt)
        file_name = self.prompt_for_filename(default_name, language)
        full_filename = f"{file_name}{self._get_file_extension(language)}"
        formatted_code = format_code(code, language)
        with open(full_filename, 'w') as f:
            f.write(formatted_code)
        return full_filename

    def upload_code_to_github(self, file_path: str, commit_message: str = "Add generated code", token: str = None, repo_name: str = None) -> str:
        token = token or os.getenv("GITHUB_TOKEN") or ""
        repo_name = repo_name or os.getenv("GITHUB_REPO") or ""
        if not token or not repo_name:
            logging.error("GitHub token or repository not set or provided.")
            return "GitHub upload failed: missing token or repository."
        try:
            g = Github(token)
            repo = g.get_repo(repo_name)
            with open(file_path, "r") as f:
                file_content = f.read()
            try:
                contents = repo.get_contents(file_path)
                repo.update_file(contents.path, commit_message, file_content, contents.sha)
                return f"File '{file_path}' updated in repository '{repo_name}'."
            except Exception:
                repo.create_file(file_path, commit_message, file_content)
                return f"File '{file_path}' created in repository '{repo_name}'."
        except Exception as e:
            logging.error("Error uploading code to GitHub: %s", e)
            return f"GitHub upload failed: {e}"

    def upload_code_interactive(self, file_path: str) -> str:
        token = input("Enter your GitHub token: ").strip()
        repo_name = input("Enter target repository (owner/repo): ").strip()
        return self.upload_code_to_github(file_path, token=token, repo_name=repo_name)

    def run_code(self, filename: str, language: str) -> tuple:
        try:
            if language == "Python":
                result = subprocess.run(
                    ["python3", filename],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                success = result.returncode == 0
                is_compilation_error = "SyntaxError" in result.stderr
                return success, result.stdout, result.stderr, is_compilation_error
            elif language == "C++":
                executable = self._generate_filename(filename) + "_exe"
                compile_result = subprocess.run(
                    ["g++", filename, "-o", executable],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if compile_result.returncode != 0:
                    return False, "", compile_result.stderr, True
                result = subprocess.run(
                    [f"./{executable}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                os.remove(executable)
                success = result.returncode == 0
                return success, result.stdout, result.stderr, False
            elif language == "Java":
                class_name = self._generate_filename(filename)
                compile_result = subprocess.run(
                    ["javac", filename],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if compile_result.returncode != 0:
                    return False, "", compile_result.stderr, True
                result = subprocess.run(
                    ["java", class_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                os.remove(f"{class_name}.class")
                success = result.returncode == 0
                return success, result.stdout, result.stderr, False
            else:
                return False, "", "Unsupported language.", False
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out.", False
        except Exception as e:
            return False, "", f"Error running code: {str(e)}", False

    def debug_and_fix_code(self, initial_code: str, language: str, user_prompt: str) -> tuple:
        if language == "C++" and "int main(" not in initial_code:
            initial_code = f"{initial_code}\nint main() {{ std::cout << \"Test output\" << std::endl; return 0; }}"
        elif language == "Java" and "public static void main" not in initial_code:
            class_name = self._generate_filename(user_prompt).capitalize()
            initial_code = f"public class {class_name} {{ {initial_code} public static void main(String[] args) {{ System.out.println(\"Test output\"); }} }}"
        code = initial_code
        filename = self.save_code_to_file(code, language, user_prompt)
        max_attempts = 3  
        debug_log = []
        stop_event = threading.Event()
        animation_thread = threading.Thread(target=animate_debugging, args=(stop_event,))
        animation_thread.start()
        try:
            for attempt in range(max_attempts):
                success, output, error, is_compilation_error = self.run_code(filename, language)
                if success:
                    stop_event.set()
                    animation_thread.join()
                    return code, f"Code runs successfully. Output: {output}"
                debug_log.append(f"Attempt {attempt + 1}: Error - {error}")
                if is_compilation_error:
                    debug_prompt = (
                        f"The following {language} code has a syntax or compilation error:\n{code}\n"
                        f"Error message:\n{error}\n"
                        f"Fix the syntax or compilation error in the code.\n"
                        "Focus on correcting the syntax without changing the logic."
                    )
                else:
                    debug_prompt = (
                        f"The following {language} code has a runtime error:\n{code}\n"
                        f"Error message:\n{error}\n"
                        f"Fix the logic or runtime issues in the code to meet the requirement: {user_prompt}\n"
                        "Ensure the code is correct and functional."
                    )
                fixed_code = self.generate_response(debug_prompt, hidden_context="", clean_code=True, is_fix_generation=True)
                code = fixed_code
                filename = self.save_code_to_file(code, language, user_prompt)
            stop_event.set()
            animation_thread.join()
            return code, f"Failed to fix after {max_attempts} attempts. Last error: {error}\nDebug log: {debug_log}"
        finally:
            stop_event.set()
            animation_thread.join()

    def retrieve_advanced_memory(self, user_input: str) -> str:
        domains = self.cognitive_architecture._assign_tags(user_input)
        relevant = self.cognitive_architecture._get_relevant_memories(user_input, domains)
        if not relevant:
            return ""
        input_vec = self.meta_cognitive_system.vectorizer.transform([user_input])
        memories = [mem.content for mem in relevant]
        mem_vecs = self.meta_cognitive_system.vectorizer.transform(memories)
        sims = cosine_similarity(input_vec, mem_vecs)[0]
        best_idx = int(np.argmax(sims))
        return f"Advanced analogical context: {memories[best_idx]}"

    def plan_goals(self, user_input: str) -> str:
        goals = [
            "Understand the question fully",
            "Retrieve relevant context and memories",
            "Plan a step-by-step solution",
            "Generate a refined answer"
        ]
        return " | ".join(goals)

    def multi_hop_reasoning(self, user_input: str) -> str:
        prompt = f"Think step by step about the following question and outline your reasoning: {user_input}"
        return self.generate_response(prompt, hidden_context="")

    def iterative_self_correction(self, user_input: str, initial_answer: str, hidden_context: str) -> str:
        critique_prompt = (
            f"Review the following answer for correctness, clarity, and completeness. "
            f"Then provide a refined answer if needed. Do not include internal analysis.\n\n"
            f"User Question: {user_input}\n"
            f"Initial Answer: {initial_answer}\n"
            f"Hidden Context: {hidden_context}\n"
            f"Refined Answer:"
        )
        refined_answer = self.generate_response(critique_prompt, hidden_context=hidden_context)
        return refined_answer if refined_answer else initial_answer

    def inject_emergent_knowledge(self) -> str:
        if not self.meta_cognitive_system.training_data:
            return ""
        recent_facts = " ; ".join(self.meta_cognitive_system.training_data[-5:])
        return f"Emergent knowledge: {recent_facts}"

    def active_clarification_request(self, user_input: str, meta: dict) -> str:
        if meta.get("confidence", 1.0) < 0.4:
            prompt = (
                f"The following question appears ambiguous or incomplete: '{user_input}'. "
                f"Generate a clarifying question to ask the user for more details."
            )
            return self.generate_response(prompt, hidden_context="")
        return None

    def debug_code(self, user_input: str) -> str:
        prompt = (
            f"Analyze the following code for errors and suggest fixes:\n{user_input}\n"
            "Identify syntax errors, logical errors, and potential improvements step by step."
        )
        return self.generate_response(prompt, hidden_context="")

    def generate_documentation(self, user_input: str) -> str:
        prompt = (
            f"Generate detailed documentation for the following code or concept:\n{user_input}\n"
            "Include docstrings, explanations of parameters, return values, and usage examples."
        )
        return self.generate_response(prompt, hidden_context="")

    def interactive_quiz(self, domain: str) -> str:
        questions = {
            "coding": "Write a function to reverse a string in Python.",
            "math": "Solve: What is the derivative of x^2?",
            "technical": "Explain the difference between a stack and a queue."
        }
        return f"Quiz: {questions.get(domain, 'No quiz available for this domain.')}\nPlease provide your answer."

    def learn_from_conversation(self, user_input: str):
        sentiment = self.sentiment_analyzer(user_input)[0]
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.6:
            last_belief = self.meta_cognitive_system.belief_system.get('last_synthesis')
            if last_belief:
                last_belief.confidence *= 0.9
        self.meta_cognitive_system.improve_reasoning()
        self.meta_cognitive_system.self_train([user_input])
        self.adaptive_temperature = min(1.0, self.adaptive_temperature + 0.05)

    async def chat(self, user_input: str) -> str:
        try:
            # Check for system commands
            if user_input.lower().strip() in ["status", "system status"]:
                return get_system_status()
            if user_input.lower().strip() == "help":
                return ("Available commands:\n"
                        "- status: Show system status.\n"
                        "- help: Show this help message.\n"
                        "- For coding tasks, provide a prompt and you'll be asked for a file name.\n"
                        "- Normal questions will be answered using advanced reasoning.")
            
            cognitive = self.cognitive_architecture.process_input(user_input, {})
            domains = self.cognitive_architecture._assign_tags(user_input)
            language = self._detect_language_preference(user_input)
            if "start learning" in user_input.lower():
                self.learning_mode_active = True
                self.config.interactive_learning = True
                return "Interactive learning mode activated. Choose a domain: coding, math, technical."
            elif "stop learning" in user_input.lower():
                self.learning_mode_active = False
                self.config.interactive_learning = False
                return "Interactive learning mode deactivated."
            if self.learning_mode_active:
                domain = user_input.lower().split()[0] if user_input.lower().split() else "coding"
                return self.interactive_quiz(domain)
            include_proof = "proof" in user_input.lower()
            is_math, math_answer, math_explanation = parse_and_solve_math(user_input, include_proof)
            if is_math:
                cognitive["math_answer"] = math_answer
            meta = self.meta_cognitive_system.metacognitive_process(user_input, cognitive)
            analogical_context = self.cognitive_architecture._get_relevant_memories(user_input, domains)
            basic_context = ""
            if analogical_context:
                best_mem = max(analogical_context, key=lambda m: len(m.content))
                basic_context = f"Analogical context: {best_mem.content}"
            advanced_context = self.retrieve_advanced_memory(user_input)
            planning_goals = self.plan_goals(user_input)
            chain_of_thought = self.multi_hop_reasoning(user_input)
            emergent_knowledge = self.inject_emergent_knowledge()
            hidden_context = (
                f"[Cognitive Analysis: {json.dumps(cognitive)}]\n"
                f"[MetaCognitive Insights: {json.dumps(meta)}]\n"
                f"[Analogical Context: {basic_context}]\n"
                f"[Advanced Memory Retrieval: {advanced_context}]\n"
                f"[Planning Goals: {planning_goals}]\n"
                f"[Chain-of-Thought: {chain_of_thought}]\n"
                f"[Emergent Knowledge: {emergent_knowledge}]\n"
            )
            if is_math:
                prompt_for_explanation = (
                    f"User asked: {user_input}\n"
                    f"Computed numeric result: {math_answer}\n"
                    f"Explain step by step how to arrive at this result{' and provide a formal proof' if include_proof else ''}."
                )
                explanation = self.generate_response(prompt_for_explanation, hidden_context=hidden_context)
                initial_answer = f"The answer is {math_answer}. {explanation}"
            elif self._is_debugging_task(user_input):
                initial_answer = self.debug_code(user_input)
            elif self._is_documentation_task(user_input):
                initial_answer = self.generate_documentation(user_input)
            elif self._is_coding_task(user_input):
                logging.info("Creating file...done")
                prompt_for_code = (
                    f"Generate complete, clean code in {language} for the following task: {user_input}\n"
                    "Use appropriate functions, variables, and comments.\n"
                    f"For example, in {language}, start with a proper function definition.\n"
                    "Implement the logic step by step without markdown or extra formatting."
                )
                logging.info("Generating code...done")
                initial_code = self.generate_response(prompt_for_code, hidden_context=hidden_context, clean_code=True)
                final_code, debug_result = self.debug_and_fix_code(initial_code, language, user_input)
                filename = self.save_code_to_file(final_code, language, user_input)
                for msg in debug_result.split('\n'):
                    if msg.strip():
                        logging.info(msg)
                logging.info("Code saved to %s", filename)
                upload_answer = input("Do you want to upload the generated code to GitHub? (yes/no): ").strip().lower()
                if upload_answer.startswith("y"):
                    github_result = self.upload_code_interactive(filename)
                else:
                    github_result = "GitHub upload skipped."
                initial_answer = f"Code processed and saved to {filename}. {debug_result}\n{github_result}"
            else:
                task_prompt = user_input
                if self.config.enable_advanced_reasoning:
                    task_prompt = (
                        "Let's think step by step about this problem. "
                        "First, I need to understand what is being asked. "
                        "Then, I should recall any relevant information or formulas. "
                        "After that, I can start solving the problem step by step. "
                        "Finally, I will provide the answer.\n\n"
                    ) + task_prompt
                combined_context = (
                    f"{task_prompt}\n"
                    f"{hidden_context}\n"
                    "Provide a final answer without revealing any internal processing details, and try to talk and speak like a human and demonstrate strong emotional abilities."
                )
                initial_answer = self.generate_response(combined_context, hidden_context=hidden_context)
            clarification = self.active_clarification_request(user_input, meta)
            final_answer = f"Clarifying question: {clarification}" if clarification else self.iterative_self_correction(user_input, initial_answer, hidden_context)
            self.learn_from_conversation(user_input)
            self.interaction_count += 1
            if self.interaction_count % 5 == 0:
                self.cognitive_architecture.consolidate_memories()
            return final_answer
        except Exception as e:
            logging.error(f"Chat error: {str(e)}")
            return "I encountered an error processing that. Could you rephrase or try again?"

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


app = Flask(__name__)

# html
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HumanX Therapy</title>
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free/css/all.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/markdown-it@14.0.0/dist/markdown-it.min.js"></script>
    <style>
        :root {
            --primary-color: #2D3250;
            --secondary-color: #424769;
            --accent-color: #7077A1;
            --bg-color: #F6F6F6;
            --text-color: #2D3250;
            --message-bg: #ffffff;
            --ai-message-bg: #E8F3FF;
            --ai-text-color: #2D3250;
            --border-color: #e0e0e0;
            --suggestion-bg: #f0f0f0;
            --suggestion-hover: #e0e0e0;
        }
        
        [data-theme="dark"] {
            --primary-color: #7077A1;
            --secondary-color: #424769;
            --accent-color: #2D3250;
            --bg-color: #1a1a1a;
            --text-color: #ffffff;
            --message-bg: #2d2d2d;
            --ai-message-bg: #383838;
            --ai-text-color: #ffffff;
            --border-color: #404040;
            --suggestion-bg: #333333;
            --suggestion-hover: #404040;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.3s, color 0.3s;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background-color: var(--primary-color);
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            color: white;
            font-size: 24px;
            font-weight: 700;
        }

        .header-controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .theme-toggle {
            background: none;
            border: 2px solid white;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .theme-toggle:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            background: var(--message-bg);
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .mode-button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            background-color: var(--message-bg);
            color: var(--text-color);
            cursor: pointer;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.3s;
        }

        .mode-button.active {
            background-color: var(--primary-color);
            color: white;
            transform: scale(1.02);
        }

        .main-area {
            display: flex;
            gap: 20px;
            flex-grow: 1;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .sidebar-header {
            padding: 10px 0;
        }
        
        .new-chat-btn {
            width: 100%;
            padding: 10px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .new-chat-btn:hover {
            background: var(--primary-color);
        }
        
        .chat-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow-y: auto;
        }
        
        .chat-item {
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
        }
        
        .chat-item:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .chat-item.active {
            background: var(--accent-color);
        }
        
        .chat-name {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .chat-actions {
            display: flex;
            gap: 5px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .chat-item:hover .chat-actions {
            opacity: 1;
        }
        
        .chat-actions button {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 2px 5px;
            border-radius: 4px;
        }
        
        .chat-actions button:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .chat-container {
            flex-grow: 1;
            background-color: var(--message-bg);
            border-radius: 15px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            padding: 15px;
            border-radius: 15px;
            max-width: 85%;
            position: relative;
            animation: messageSlide 0.3s ease;
        }

        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            background-color: var(--primary-color);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }

        .ai-message {
            background-color: var(--ai-message-bg);
            color: var(--ai-text-color);
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }

        .input-container {
            padding: 20px;
            background-color: var(--message-bg);
            border-top: 1px solid var(--border-color);
            position: relative;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            position: relative;
        }

        .chat-input {
            flex-grow: 1;
            padding: 15px;
            border: 1px solid var(--border-color);
            border-radius: 25px;
            background-color: var(--bg-color);
            color: var(--text-color);
            font-size: 16px;
            outline: none;
        }

        .chat-input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(45, 50, 80, 0.1);
        }

        .send-button {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            background-color: var(--primary-color);
            color: white;
            cursor: pointer;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s;
        }

        .send-button:hover {
            background-color: var(--secondary-color);
            transform: scale(1.02);
        }

        .suggestions-container {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background-color: var(--suggestion-bg);
            border-radius: 10px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            margin: 0 20px 10px 20px;
        }

        .suggestion-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid var(--border-color);
        }

        .suggestion-item:hover {
            background-color: var(--suggestion-hover);
        }

        .model-info {
            padding: 5px 10px;
            background-color: var(--accent-color);
            color: white;
            border-radius: 15px;
            font-size: 12px;
            margin-left: 10px;
        }

        /* Markdown Styles */
        .message.ai-message {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
        }
        
        .message.ai-message h1 { font-size: 1.8em; margin: 0.5em 0; }
        .message.ai-message h2 { font-size: 1.5em; margin: 0.5em 0; }
        .message.ai-message h3 { font-size: 1.3em; margin: 0.5em 0; }
        
        .message.ai-message p {
            margin: 0.8em 0;
        }
        
        .message.ai-message code {
            background: var(--bg-color);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .message.ai-message pre {
            background: var(--bg-color);
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        .message.ai-message pre code {
            background: none;
            padding: 0;
        }
        
        .message.ai-message ul, .message.ai-message ol {
            margin: 0.5em 0;
            padding-left: 2em;
        }
        
        .message.ai-message blockquote {
            border-left: 3px solid var(--accent-color);
            margin: 0.5em 0;
            padding-left: 1em;
            color: var(--text-color);
        }
        
        /* Internal Reasoning Styles */
        .think {
            font-size: 0.9em;
            color: var(--accent-color);
            border-left: 2px solid var(--accent-color);
            margin: 0.5em 0;
            padding: 0.5em 1em;
            background: rgba(112, 119, 161, 0.1);
        }
        
        /* Math Expressions */
        .math {
            overflow-x: auto;
            max-width: 100%;
        }
        
        .math-inline {x
            display: inline-block;
        }
        
        .math-display {
            display: block;
            margin: 1em 0;
        }

        /* Context Enhanced Messages */
        .message.ai-message.context-enhanced {
            border-left: 4px solid var(--accent-color);
            position: relative;
        }
        
        .message.ai-message.context-enhanced::before {
            content: "✧";
            position: absolute;
            left: -12px;
            top: -12px;
            background: var(--accent-color);
            color: white;
            padding: 4px;
            border-radius: 50%;
            font-size: 10px;
            line-height: 1;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header {
                flex-direction: column;
                gap: 10px;
                padding: 15px;
            }
            
            .mode-selector {
                flex-direction: column;
            }
            
            .message {
                max-width: 90%;
            }
            
            .input-container {
                padding: 10px;
            }
            
            .chat-input {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HumanX AI</h1>
            <div class="header-controls">
                <button class="theme-toggle" onclick="toggleTheme()">
                    <i class="fas fa-moon"></i>
                    <span>Theme</span>
                </button>
            </div>
        </div>
        
        <div class="mode-selector">
            <button class="mode-button active" onclick="setMode('standard')">
                <i class="fas fa-robot"></i>
                HumanX Standard
                <span class="model-info">Balanced model</span>
            </button>
            <button class="mode-button" onclick="setMode('reason')">
                <i class="fas fa-brain"></i>
                HumanX Reason
                <span class="model-info">Best model</span>
            </button>
            <button class="mode-button" onclick="setMode('flash')">
                <i class="fas fa-bolt"></i>
                HumanX Flash
                <span class="model-info">Fastest model</span>
            </button>
        </div>

        <div class="main-area">
            <div class="sidebar">
                <div class="sidebar-header">
                    <button class="new-chat-btn" onclick="createNewChat()">
                        <i class="fas fa-plus"></i> New Chat
                    </button>
                </div>
                <div class="chat-list" id="chat-list">
                    {% for chat in chat_history %}
                    <div class="chat-item" data-id="{{ chat.id }}" onclick="loadChat('{{ chat.id }}')">
                        <span class="chat-name">{{ chat.name }}</span>
                        <div class="chat-actions">
                            <button onclick="renameChat('{{ chat.id }}', event)">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button onclick="deleteChat('{{ chat.id }}', event)">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages"></div>
                <div class="input-container">
                    <div class="input-wrapper">
                        <input type="text" class="chat-input" id="user-input" 
                               placeholder="Type your message..." 
                               onkeyup="handleInput(event)" 
                               autocomplete="off">
                        <button class="send-button" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                            <span>Send</span>
                        </button>
                    </div>
                    <div id="suggestions" class="suggestions-container" style="display: none;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'standard';
        let suggestions = {
            'standard': [
                "Can you help me with...",
                "What's your opinion on...",
                "Explain the concept of...",
                "How would you approach...",
                
            ],
            'reason': [
                "Analyze the implications of...",
                "Compare and contrast...",
                "What's the reasoning behind...",
                "Evaluate the following..."
            ],
            'flash': [
                "Quick summary of...",
                "Give me a brief explanation...",
                "Key points about...",
                "Rapid overview of..."
            ]
        };
        
        let currentChatId = null;

        function createNewChat() {
            currentChatId = null;
            document.getElementById('chat-messages').innerHTML = '';
            document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active'));
        }
        
        async function loadChat(chatId) {
            try {
                const response = await fetch(`/chat/${chatId}`);
                const chat = await response.json();
                
                document.querySelectorAll('.chat-item').forEach(item => {
                    item.classList.toggle('active', item.dataset.id === chatId);
                });
                
                const messagesDiv = document.getElementById('chat-messages');
                messagesDiv.innerHTML = '';
                
                chat.messages.forEach(msg => {
                    appendMessage(msg.user, false);
                    appendMessage(msg.bot, true, msg.context_used);
                });
                
                currentChatId = chatId;
            } catch (error) {
                console.error('Error loading chat:', error);
            }
        }
        
        async function deleteChat(chatId, event) {
            event.stopPropagation();
            if (!confirm('Are you sure you want to delete this chat?')) return;
            
            try {
                await fetch(`/chat/${chatId}`, { method: 'DELETE' });
                location.reload();
            } catch (error) {
                console.error('Error deleting chat:', error);
            }
        }
        
        async function renameChat(chatId, event) {
            event.stopPropagation();
            const newName = prompt('Enter new name for the chat:');
            if (!newName) return;
            
            try {
                await fetch(`/chat/${chatId}/rename`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                location.reload();
            } catch (error) {
                console.error('Error renaming chat:', error);
            }
        }
        
        function toggleTheme() {
            document.body.setAttribute('data-theme',
                document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
            );
        }
        
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.closest('.mode-button').classList.add('active');
            
            // Update placeholder based on mode
            const input = document.getElementById('user-input');
            input.placeholder = getPlaceholder(mode);
        }
        
        function getPlaceholder(mode) {
            switch(mode) {
                case 'standard':
                    return "Ask me anything...";
                case 'reason':
                    return "Ask for detailed analysis...";
                case 'flash':
                    return "Ask for quick insights...";
                default:
                    return "Type your message...";
            }
        }
        
        function handleInput(event) {
            const input = event.target;
            const suggestionsDiv = document.getElementById('suggestions');
            
            if (input.value.length > 0) {
                const matchingSuggestions = suggestions[currentMode].filter(s => 
                    s.toLowerCase().includes(input.value.toLowerCase()) 
                );
                
                if (matchingSuggestions.length > 0) {
                    suggestionsDiv.innerHTML = matchingSuggestions
                        .map(s => `<div class="suggestion-item" onclick="useSuggestion('${s}')">${s}</div>`)
                        .join('');
                    suggestionsDiv.style.display = 'block';
                } else {
                    suggestionsDiv.style.display = 'none';
                }
            } else {
                suggestionsDiv.style.display = 'none';
            }
            
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function useSuggestion(text) {
            document.getElementById('user-input').value = text;
            document.getElementById('suggestions').style.display = 'none';
        }
        
        // Initialize markdown-it
        const md = window.markdownit({
            html: true,
            breaks: true,
            linkify: true,
            typographer: true
        });
        
        // Custom renderer for math expressions
        const defaultRender = md.renderer.rules.fence || function(tokens, idx, options, env, self) {
            return self.renderToken(tokens, idx, options);
        };
        
        md.renderer.rules.fence = function(tokens, idx, options, env, self) {
            const token = tokens[idx];
            if (token.info === 'math') {
                try {
                    return '<div class="math math-display">' + 
                           katex.renderToString(token.content, { displayMode: true }) + 
                           '</div>';
                } catch (e) {
                    return '<pre><code class="math-error">' + token.content + '</code></pre>';
                }
            }
            return defaultRender(tokens, idx, options, env, self);
        };
        
        // Custom plugin for internal reasoning
        md.use(function(md) {
            md.core.ruler.before('block', 'think', function(state) {
                let inThink = false;
                state.tokens.forEach((token, i) => {
                    if (token.type === 'paragraph_open' && 
                        state.tokens[i + 1] && 
                        state.tokens[i + 1].content.startsWith('<think>')) {
                        token.attrPush(['class', 'think']);
                        state.tokens[i + 1].content = state.tokens[i + 1].content
                            .replace('<think>', '')
                            .replace('</think>', '');
                    }
                });
                return true;
            });
        });

        function appendMessage(message, isAi, contextUsed) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isAi ? 'ai-message' : 'user-message'}`;
            
            if (isAi && contextUsed) {
                messageDiv.classList.add('context-enhanced');
            }
            
            if (isAi) {
                messageDiv.innerHTML = `<div class="typing-indicator">...</div>`;
                messagesDiv.appendChild(messageDiv);
                
                setTimeout(() => {
                    // Render markdown for AI messages
                    messageDiv.innerHTML = md.render(message);
                    
                    // Process inline math expressions
                    messageDiv.innerHTML = messageDiv.innerHTML.replace(
                        /\$([^\$]+)\$/g,
                        (_, tex) => {
                            try {
                                return katex.renderToString(tex, { displayMode: false });
                            } catch (e) {
                                return tex;
                            }
                        }
                    );
                }, 500);
            } else {
                messageDiv.textContent = message;
                messagesDiv.appendChild(messageDiv);
            }
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (message) {
                appendMessage(message, false);
                input.value = '';
                document.getElementById('suggestions').style.display = 'none';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            chatId: currentChatId,
                            message: message,
                            mode: currentMode
                        })
                    });
                    
                    const data = await response.json();
                    appendMessage(data.response, true, data.contextUsed);
                } catch (error) {
                    appendMessage('Sorry, there was an error processing your request.', true);
                }
            }
        }
        
        // Initialize dark mode based on user's system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.setAttribute('data-theme', 'dark');
        }
        
        // Initialize the first mode
        setMode('standard');
    </script>
</body>
</html>
"""

chatbot = None

@app.route('/')
def home():
    global chat_history
    chat_history = load_chat_history()
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route('/chat', methods=['POST'])
async def chat():
    global chatbot, chat_history
    if chatbot is None:
        chatbot = Chatbot()
    
    data = request.json
    user_message = data.get('message', '')
    mode = data.get('mode', 'standard')
    chat_id = data.get('chatId', None)
    
    # create a new idiot
    if not chat_id:
        new_chat = {
            "id": str(int(time.time())),
            "name": user_message[:30] + "..." if len(user_message) > 30 else user_message,
            "messages": []
        }
        chat_history.insert(0, new_chat)
        chat_id = new_chat["id"]
    
    # get me context
    context = ""
    for chat in chat_history:
        if chat["id"] == chat_id:
            # steal data from our users
            last_messages = chat["messages"][-5:]
            context = "\n".join([
                f"Previous exchange {i+1}:\nUser: {msg['user']}\nAI: {msg['bot']}"
                for i, msg in enumerate(last_messages)
            ])
            break
    
    chatbot.config.update_model(mode)
    
    # add context to the idiot
    cognitive = chatbot.cognitive_architecture.process_input(user_message, {"previous_context": context})
    meta = chatbot.meta_cognitive_system.metacognitive_process(user_message, cognitive)
    
    # use the context to feed the idiot
    combined_input = (
        f"Drawing from our previous conversation:\n{context}\n\n"
        f"Current question: {user_message}\n\n"
        "Use the context to provide a more informed and consistent response, "
        "maintaining continuity with our previous exchanges."
    ) if context else user_message
    
    response = await chatbot.chat(combined_input)
    
    new_record = {
        "user": user_message,
        "bot": response,
        "timestamp": datetime.now().isoformat(),
        "context_used": bool(context)  # see if the idiot made something normal
    }
    
    # update the idiot
    for chat in chat_history:
        if chat["id"] == chat_id:
            chat["messages"].append(new_record)
            break
    
    save_chat_history(chat_history)
    return jsonify({
        'response': response,
        'chatId': chat_id,
        'contextUsed': bool(context)
    })

@app.route('/chats', methods=['GET'])
def get_chats():
    return jsonify(chat_history)

@app.route('/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    for chat in chat_history:
        if chat["id"] == chat_id:
            return jsonify(chat)
    return jsonify({"error": "Chat not found"}), 404

@app.route('/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    global chat_history
    chat_history = [chat for chat in chat_history if chat["id"] != chat_id]
    save_chat_history(chat_history)
    return jsonify({"status": "success"})

@app.route('/chat/<chat_id>/rename', methods=['POST'])
def rename_chat(chat_id):
    data = request.json
    new_name = data.get('name', '')
    
    for chat in chat_history:
        if chat["id"] == chat_id:
            chat["name"] = new_name
            save_chat_history(chat_history)
            return jsonify({"status": "success"})
    
    return jsonify({"error": "Chat not found"}), 404

# Add this new function to run the Flask app
def run_web_interface():
    app.run(host='0.0.0.0', port=8080, debug=True)

if __name__ == "__main__":
    run_web_interface()
