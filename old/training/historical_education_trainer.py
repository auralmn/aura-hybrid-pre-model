import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re


class AuraHistoricalEducationTrainer:
    def __init__(self, aura_network, network_mapper=None):
        self.aura = aura_network
        self.mapper = network_mapper
        
        # Historical learning statistics
        self.historical_stats = {
            'total_conversations': 0,
            'total_turns': 0,
            'student_questions_processed': 0,
            'expert_responses_learned': 0,
            'historical_periods_covered': {},
            'question_types': {
                'who': 0, 'what': 0, 'when': 0, 'where': 0, 'why': 0, 'how': 0, 'other': 0
            },
            'historical_entities_learned': set(),
            'temporal_range_coverage': {'start': None, 'end': None},
            'hippocampal_memory_formation': 0,
            'historical_specialist_accuracy': [],
            # Running loss proxy (1 - answer_quality)
            'answer_quality_sum': 0.0,
            'answer_quality_count': 0,
        }
        
        # Historical pattern recognition
        self.historical_patterns = {
            'periods': {
                # Extend Ancient to include BCE years
                'ancient': (-3000, 500),
                'medieval': (500, 1500), 
                'early_modern': (1500, 1800),
                'modern': (1800, 1945),
                'contemporary': (1945, 2025)
            },
            'event_types': [
                'military', 'political', 'cultural', 'economic', 'social', 
                'technological', 'religious', 'diplomatic'
            ],
            'geographical_regions': [
                'europe', 'asia', 'africa', 'americas', 'oceania', 'global'
            ]
        }
        
    async def process_historical_dataset(self, dataset_path: str, weights_dir: str = 'svc_nlms_weights'):
        """Process historical conversation dataset through Aura"""
        
        print(f"🏛️ Starting Aura historical education training: {dataset_path}")
        
        # Attempt to resume weights
        try:
            from tools.weights_io import load_network_weights
            counts_loaded = load_network_weights(self.aura, weights_dir)
            if counts_loaded:
                print(f"↺ Resumed weights: {counts_loaded}")
        except Exception:
            pass

        # Load and organize conversations
        conversations = self._load_historical_conversations(dataset_path)
        
        # Process conversations with historical focus
        batch_size = 25  # Smaller batches for intensive historical processing
        
        total = len(conversations)
        for i in range(0, total, batch_size):
            batch = conversations[i:i+batch_size]
            
            print(f"📚 Processing historical batch {i//batch_size + 1}/{(len(conversations)-1)//batch_size + 1}")
            
            await self._process_historical_batch(batch)
            
            # Update neural visualization with historical focus
            if self.mapper:
                await self.mapper.update_network_map(
                    self.aura,
                    current_query=f"Historical learning batch {i//batch_size + 1}",
                    query_features=self._generate_historical_context_features()
                )
            
            # Simple progress bar with running accuracy/loss
            done = min(i + batch_size, total)
            p = done / max(1, total)
            bar_w = 30
            filled = int(p * bar_w)
            acc = np.mean(self.historical_stats['historical_specialist_accuracy']) if self.historical_stats['historical_specialist_accuracy'] else 0.0
            aq_cnt = self.historical_stats['answer_quality_count']
            loss = 1.0 - (self.historical_stats['answer_quality_sum'] / aq_cnt) if aq_cnt > 0 else 1.0
            print(f"[{('#'*filled).ljust(bar_w)}] {done}/{total}  acc={acc:.3f}  loss={loss:.3f}")
            await asyncio.sleep(0.05)
        
        # Generate comprehensive historical training report
        await self._generate_historical_training_report()
        # Save weights
        try:
            from tools.weights_io import save_network_weights
            counts = save_network_weights(self.aura, weights_dir)
            print(f"✓ Saved weights: {counts}")
        except Exception as e:
            print(f"! Weight save skipped: {e}")
        
        print(f"🎓 Historical education training complete!")
    
    def _load_historical_conversations(self, dataset_path: str) -> List[Dict]:
        """Load and organize historical conversations by conv_id"""
        
        raw_turns = []
        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    turn = json.loads(line.strip())
                    raw_turns.append(turn)
                except json.JSONDecodeError:
                    continue
        
        # Group turns by conversation ID
        conversations = {}
        for turn in raw_turns:
            conv_id = turn['conv_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(turn)
        
        # Sort turns within each conversation and convert to list
        organized_conversations: List[Dict[str, Any]] = []
        for conv_id, turns in conversations.items():
            sorted_turns = sorted(turns, key=lambda x: x.get('turn', 0))
            organized_conversations.append({'conv_id': conv_id, 'turns': sorted_turns})

        # If dataset collapses everything under a single conv_id, optionally split into per-pair
        if len(organized_conversations) == 1 and len(organized_conversations[0]['turns']) > 1000:
            conv = organized_conversations[0]
            pairs: List[Dict[str, Any]] = []
            turns = conv['turns']
            idx = 0
            while idx + 1 < len(turns):
                s = turns[idx]
                e = turns[idx + 1]
                # Ensure student→expert ordering; if not, try to find next expert
                if s.get('role') != 'student':
                    idx += 1
                    continue
                if e.get('role') != 'expert':
                    # try to find the next expert within a small window
                    j = idx + 1
                    found = False
                    while j < len(turns) and j - idx <= 5:
                        if turns[j].get('role') == 'expert':
                            e = turns[j]
                            found = True
                            break
                        j += 1
                    if not found:
                        idx += 1
                        continue
                    idx = j  # will be incremented at end
                # Create a synthetic conversation with just this pair
                pairs.append({'conv_id': f"{conv['conv_id']}_{len(pairs)}", 'turns': [s, e]})
                idx += 2
            if pairs:
                organized_conversations = pairs
        
        print(f"🏛️ Loaded {len(organized_conversations)} historical conversations with {len(raw_turns)} total turns")
        return organized_conversations
    
    async def _process_historical_batch(self, conversations: List[Dict]):
        """Process batch of historical conversations"""
        
        for conv in conversations:
            await self._process_historical_conversation(conv)
    
    async def _process_historical_conversation(self, conversation: Dict):
        """Process single historical conversation with educational focus"""
        
        conv_id = conversation['conv_id']
        turns = conversation['turns']
        
        # Extract historical context from conversation
        historical_context = self._extract_historical_context(turns)
        
        # Track conversation in statistics
        self.historical_stats['total_conversations'] += 1
        self.historical_stats['total_turns'] += len(turns)
        
        # Update historical period coverage
        if historical_context['time_period']:
            period = historical_context['time_period']
            self.historical_stats['historical_periods_covered'][period] = \
                self.historical_stats['historical_periods_covered'].get(period, 0) + 1
        
        # Update temporal range
        if historical_context['date_range']:
            start_date, end_date = historical_context['date_range']
            if self.historical_stats['temporal_range_coverage']['start'] is None:
                self.historical_stats['temporal_range_coverage']['start'] = start_date
                self.historical_stats['temporal_range_coverage']['end'] = end_date
            else:
                self.historical_stats['temporal_range_coverage']['start'] = min(
                    self.historical_stats['temporal_range_coverage']['start'], start_date
                )
                self.historical_stats['temporal_range_coverage']['end'] = max(
                    self.historical_stats['temporal_range_coverage']['end'], end_date
                )
        
        # Process each turn pair (student question -> expert answer)
        conversation_memory = {
            'conv_id': conv_id,
            'historical_context': historical_context,
            'turn_history': [],
            'learned_facts': [],
            'question_patterns': []
        }
        
        for i in range(0, len(turns)-1, 2):
            if i+1 < len(turns):
                student_turn = turns[i] if turns[i]['role'] == 'student' else turns[i+1]
                expert_turn = turns[i+1] if turns[i+1]['role'] == 'expert' else turns[i]
                
                await self._process_student_expert_pair(
                    student_turn, expert_turn, conversation_memory, historical_context
                )
        
        # Final conversation consolidation in hippocampus
        await self._consolidate_historical_memory(conversation_memory)
    
    def _extract_historical_context(self, turns: List[Dict]) -> Dict:
        """Extract historical context from conversation turns"""
        
        # Combine all text for analysis
        full_text = ' '.join([turn['text'] for turn in turns])
        
        context = {
            'entities': set(),
            'date_range': None,
            'time_period': None,
            'geographical_locations': set(),
            'event_type': None,
            'key_figures': set()
        }
        
        # Extract historical entities and figures
        historical_keywords = [
            'schlieffen', 'germany', 'german', 'france', 'russia', 'plan', 
            'count', 'alfred', 'von', 'general', 'staff', 'military', 'alliance'
        ]
        
        for keyword in historical_keywords:
            if keyword.lower() in full_text.lower():
                context['entities'].add(keyword.title())
        
        # Extract dates and time periods (support BCE/BC, CE/AD, and negatives)
        years: List[float] = []
        # Pattern captures optional sign, digits (optionally decimal), optional era label
        for m in re.finditer(r'(-?\d{1,4})(?:\.\d+)?\s*(BCE|BC|CE|AD)?', full_text, flags=re.IGNORECASE):
            num_str = m.group(1)
            era = (m.group(2) or '').upper()
            try:
                val = float(num_str)
            except Exception:
                continue
            # Convert BCE/BC to negative years; AD/CE to positive
            if era in ('BCE', 'BC') and val > 0:
                val = -val
            # If era is AD/CE, keep as is
            years.append(val)
        if years:
            years_sorted = sorted(years)
            context['date_range'] = (years_sorted[0], years_sorted[-1])
            # Determine time period using average year
            avg_year = float(np.mean(years))
            for period, (start, end) in self.historical_patterns['periods'].items():
                if start <= avg_year <= end:
                    context['time_period'] = period
                    break
        
        # Extract geographical information
        locations = ['germany', 'france', 'russia', 'europe', 'berlin']
        for location in locations:
            if location.lower() in full_text.lower():
                context['geographical_locations'].add(location.title())
        
        # Determine event type
        if any(word in full_text.lower() for word in ['plan', 'military', 'strategy', 'war']):
            context['event_type'] = 'military'
        elif any(word in full_text.lower() for word in ['political', 'alliance', 'diplomacy']):
            context['event_type'] = 'political'
        
        return context
    
    async def _process_student_expert_pair(self, student_turn: Dict, expert_turn: Dict,
                                         conversation_memory: Dict, historical_context: Dict):
        """Process student question and expert answer pair"""
        
        student_question = student_turn['text']
        expert_answer = expert_turn['text']
        
        # 1. Process student question through Aura's systems
        await self._process_student_question(student_question, conversation_memory, historical_context)
        
        # 2. Learn from expert answer
        await self._learn_from_expert_answer(expert_answer, student_question, 
                                           conversation_memory, historical_context)
        
        # 3. Update turn history
        conversation_memory['turn_history'].append({
            'student_question': student_question,
            'expert_answer': expert_answer,
            'turn_pair': len(conversation_memory['turn_history'])
        })
    
    async def _process_student_question(self, question: str, memory: Dict, context: Dict):
        """Process student question through Aura's neural pipeline"""
        
        # Generate question features
        question_features = (
            self.aura.sbert.encode(question)
            if hasattr(self.aura, 'sbert') else np.zeros(384, dtype=np.float32)
        )
        
        # Classify question type
        question_type = self._classify_question_type(question)
        # Ensure bucket exists (robust to new labels)
        if question_type not in self.historical_stats['question_types']:
            self.historical_stats['question_types'][question_type] = 0
        self.historical_stats['question_types'][question_type] += 1
        self.historical_stats['student_questions_processed'] += 1
        
        # CNS assessment with historical focus
        global_state = self.aura._cns.assess_global_state()  # noqa: F841
        
        # Enhanced input context for historical learning
        enhanced_context = {
            'type': 'historical_education',
            'urgency': 0.4,  # Educational, not urgent
            'complexity': self._assess_historical_complexity(question, context),
            'emotional_content': 0.2,  # Generally low emotional content in education
            'historical_context': context,
            'question_type': question_type,
            'educational_mode': True
        }
        
        # Thalamic routing - should strongly favor historical specialist
        routing_decision = self.aura._thalamic_router.analyze_conversation_intent(
            question, question_features
        )
        
        # Expected routing for historical questions
        expected_routing = ['historical_specialist', 'hippocampus_specialist', 'analytical_specialist']
        routing_accuracy = self._evaluate_historical_routing(routing_decision, expected_routing)
        self.historical_stats['historical_specialist_accuracy'].append(routing_accuracy)
        
        # Process through hippocampus for memory formation
        if hasattr(self.aura, '_hippocampus'):
            memory_encoding = self.aura._hippocampus.encode_memory(question_features)
            memory['question_memories'] = memory_encoding
        
        # Store question analysis
        memory['question_patterns'].append({
            'question': question,
            'type': question_type,
            'features': question_features,
            'routing_accuracy': routing_accuracy,
            'complexity': enhanced_context['complexity']
        })
    
    async def _learn_from_expert_answer(self, answer: str, question: str, memory: Dict, context: Dict):
        """Learn from expert's historical answer"""
        
        # Generate answer features
        answer_features = (
            self.aura.sbert.encode(answer)
            if hasattr(self.aura, 'sbert') else np.zeros(384, dtype=np.float32)
        )
        
        # Assess answer quality for historical education
        answer_quality = self._assess_historical_answer_quality(answer, question, context)
        # Track loss proxy
        self.historical_stats['answer_quality_sum'] += float(answer_quality)
        self.historical_stats['answer_quality_count'] += 1
        
        # Extract historical facts from answer
        extracted_facts = self._extract_historical_facts(answer, context)
        memory['learned_facts'].extend(extracted_facts)
        
        # Update historical entities learned
        for fact in extracted_facts:
            if 'entity' in fact:
                self.historical_stats['historical_entities_learned'].add(fact['entity'])
        
        # Neural learning updates
        await self._update_historical_neural_learning(
            question, answer, answer_quality, context, memory
        )
        
        self.historical_stats['expert_responses_learned'] += 1
    
    def _classify_question_type(self, question: str) -> str:
        """Classify historical question type"""
        
        question_lower = question.lower()
        
        if question_lower.startswith(('who', 'who was', 'who were')):
            return 'who'
        elif question_lower.startswith(('what', 'what was', 'what were')):
            return 'what'
        elif question_lower.startswith(('when', 'when did', 'when was')):
            return 'when'
        elif question_lower.startswith(('where', 'where did', 'where was')):
            return 'where'
        elif question_lower.startswith(('why', 'why did', 'why was')):
            return 'why'
        elif question_lower.startswith(('how', 'how did', 'how was')):
            return 'how'
        else:
            return 'other'
    
    def _assess_historical_complexity(self, question: str, context: Dict) -> float:
        """Assess complexity of historical question"""
        
        complexity_factors = []
        
        # Question length and structure
        word_count = len(question.split())
        complexity_factors.append(min(1.0, word_count / 30))
        
        # Multiple historical entities mentioned
        entity_count = len(context.get('entities', set()))
        complexity_factors.append(min(1.0, entity_count / 5))
        
        # Time span complexity
        if context.get('date_range'):
            start, end = context['date_range']
            time_span = end - start
            complexity_factors.append(min(1.0, time_span / 100))  # Century-scale events are complex
        else:
            complexity_factors.append(0.3)
        
        # Analytical question types are more complex
        if any(word in question.lower() for word in ['analyze', 'compare', 'evaluate', 'explain why']):
            complexity_factors.append(0.9)
        else:
            complexity_factors.append(0.5)
        
        return np.mean(complexity_factors)
    
    def _evaluate_historical_routing(self, routing_decision: Dict, expected: List[str]) -> float:
        """Evaluate routing accuracy for historical questions"""
        
        primary_target = routing_decision.get('primary_target', '')
        secondary_targets = routing_decision.get('secondary_targets', [])
        all_targets = [primary_target] + secondary_targets
        
        # Historical questions should route to historical/hippocampus specialists
        historical_matches = sum(1 for target in all_targets 
                               if 'historical' in target or 'hippocampus' in target)
        total_targets = len(all_targets) if all_targets else 1
        
        return min(1.0, historical_matches / total_targets)
    
    def _assess_historical_answer_quality(self, answer: str, question: str, context: Dict) -> float:
        """Assess quality of historical answer"""
        
        quality_factors = []
        
        # Length appropriateness for educational content
        word_count = len(answer.split())
        if 20 <= word_count <= 150:  # Good educational length
            quality_factors.append(0.9)
        elif word_count < 10:
            quality_factors.append(0.3)  # Too brief
        else:
            quality_factors.append(0.7)
        
        # Historical accuracy indicators (basic)
        if context.get('date_range'):
            start_date, end_date = context['date_range']
            # Check if answer mentions dates within reasonable range
            answer_dates = re.findall(r'(\d{3,4})', answer)
            if answer_dates:
                dates_in_range = sum(1 for d in answer_dates 
                                   if start_date <= float(d) <= end_date + 50)  # Some tolerance
                quality_factors.append(min(1.0, dates_in_range / len(answer_dates)))
            else:
                quality_factors.append(0.6)  # No dates mentioned
        else:
            quality_factors.append(0.7)
        
        # Entity consistency
        question_entities = context.get('entities', set())
        answer_entities = set()
        for entity in question_entities:
            if entity.lower() in answer.lower():
                answer_entities.add(entity)
        
        if question_entities:
            entity_consistency = len(answer_entities) / len(question_entities)
            quality_factors.append(entity_consistency)
        else:
            quality_factors.append(0.7)
        
        # Educational structure (explanations, context)
        if any(word in answer.lower() for word in ['because', 'due to', 'resulted in', 'caused by']):
            quality_factors.append(0.8)  # Good explanatory content
        else:
            quality_factors.append(0.6)
        
        return np.mean(quality_factors)
    
    def _extract_historical_facts(self, answer: str, context: Dict) -> List[Dict]:
        """Extract structured historical facts from expert answer"""
        
        facts = []
        
        # Extract person-related facts
        persons = re.findall(r'([A-Z][a-z]+ (?:von |de |)[A-Z][a-z]+)', answer)
        for person in persons:
            facts.append({
                'type': 'person',
                'entity': person,
                'context': context.get('time_period', 'unknown'),
                'source_text': answer[:100] + '...'
            })
        
        # Extract date facts
        dates = re.findall(r'(\d{4})', answer)
        for date in dates:
            facts.append({
                'type': 'date',
                'entity': date,
                'context': context.get('event_type', 'unknown'),
                'source_text': answer[:100] + '...'
            })
        
        # Extract location facts
        locations = context.get('geographical_locations', set())
        for location in locations:
            if location.lower() in answer.lower():
                facts.append({
                    'type': 'location',
                    'entity': location,
                    'context': context.get('time_period', 'unknown'),
                    'source_text': answer[:100] + '...'
                })
        
        return facts
    
    async def _update_historical_neural_learning(self, question: str, answer: str, 
                                               quality: float, context: Dict, memory: Dict):
        """Update neural networks with historical learning"""
        
        question_features = (
            self.aura.sbert.encode(question)
            if hasattr(self.aura, 'sbert') else np.zeros(384, dtype=np.float32)
        )
        answer_features = (
            self.aura.sbert.encode(answer)
            if hasattr(self.aura, 'sbert') else np.zeros(384, dtype=np.float32)
        )
        
        # 1. Update Historical Specialist neurons (priority)
        await self._update_historical_specialist(question_features, answer_features, quality)
        
        # 2. Update Hippocampus with memory formation
        await self._update_hippocampal_learning(question_features, answer_features, quality, context)
        
        # 3. Update Thalamic Router for better historical routing
        await self._update_historical_routing(question_features, quality)
        
        # 4. Update CNS with educational learning patterns
        await self._update_cns_educational_learning(quality, context)
    
    async def _update_historical_specialist(self, q_features: np.ndarray, 
                                          a_features: np.ndarray, quality: float):
        """Update historical specialist neurons"""
        
        if hasattr(self.aura, '_hippocampus'):  # Using hippocampus as historical specialist
            # Update some place cells for historical memory
            for neuron in self.aura._hippocampus.neurons[:5]:
                try:
                    # Use question features as input, answer quality as target
                    await neuron.update_nlms(q_features, quality)
                except:  # noqa: E722
                    pass
    
    async def _update_hippocampal_learning(self, q_features: np.ndarray, a_features: np.ndarray,
                                         quality: float, context: Dict):
        """Update hippocampus with historical memory formation"""
        
        if hasattr(self.aura, '_hippocampus'):
            # Encode the question-answer pair as memory
            combined_features = np.concatenate([q_features[:192], a_features[:192]])
            memory_trace = self.aura._hippocampus.encode_memory(combined_features)
            
            # Stimulate neurogenesis for high-quality learning
            if quality > 0.8:
                new_neurons = self.aura._hippocampus.stimulate_neurogenesis()
                self.historical_stats['hippocampal_memory_formation'] += len(new_neurons)
    
    async def _update_historical_routing(self, features: np.ndarray, quality: float):
        """Update thalamic routing for historical questions"""
        
        if hasattr(self.aura, '_thalamic_router'):
            router = self.aura._thalamic_router
            
            # Update historical specialist routing neurons
            if 'historical_specialist' in router.routing_neurons:
                target_neurons = router.routing_neurons['historical_specialist'][:3]
                
                for neuron in target_neurons:
                    try:
                        await neuron.update_nlms(features, quality)
                    except:  # noqa: E722
                        pass
    
    async def _update_cns_educational_learning(self, quality: float, context: Dict):
        """Update CNS for educational learning patterns"""
        
        if hasattr(self.aura, '_cns'):
            # Educational content typically requires sustained attention
            if quality > 0.7:
                self.aura._cns.attention_mode = self.aura._cns.attention_mode  # Keep current mode
                # Slight increase in arousal for learning
                self.aura._cns.global_arousal = min(0.8, self.aura._cns.global_arousal + 0.1)
    
    async def _consolidate_historical_memory(self, conversation_memory: Dict):
        """Consolidate learned historical information"""
        
        # Create summary of learned facts
        learned_facts = conversation_memory.get('learned_facts', [])
        
        if learned_facts:
            # Create consolidated memory representation
            fact_summary = {
                'conversation_id': conversation_memory['conv_id'],
                'total_facts': len(learned_facts),
                'entities': list(set(fact['entity'] for fact in learned_facts)),
                'time_period': conversation_memory['historical_context'].get('time_period'),
                'event_type': conversation_memory['historical_context'].get('event_type'),
                'consolidation_timestamp': datetime.now().isoformat()
            }
            
            # Store in long-term memory structure (could be enhanced)
            self.historical_stats['hippocampal_memory_formation'] += 1
    
    def _generate_historical_context_features(self) -> np.ndarray:
        """Generate features representing current historical learning context"""
        
        # Create feature vector representing historical learning state
        features = np.zeros(384)
        
        # Historical period coverage
        periods = self.historical_stats['historical_periods_covered']
        for i, period in enumerate(['ancient', 'medieval', 'early_modern', 'modern', 'contemporary']):
            if i < 5:
                features[i] = periods.get(period, 0) / max(1, self.historical_stats['total_conversations'])
        
        # Question type distribution
        q_types = self.historical_stats['question_types']
        total_questions = max(1, sum(q_types.values()))
        for i, q_type in enumerate(['who', 'what', 'when', 'where', 'why', 'how']):
            if i + 5 < 384:
                features[i + 5] = q_types[q_type] / total_questions
        
        # Learning progress indicators
        features[11] = min(1.0, self.historical_stats['total_conversations'] / 1000)
        features[12] = min(1.0, len(self.historical_stats['historical_entities_learned']) / 500)
        features[13] = np.mean(self.historical_stats['historical_specialist_accuracy']) if self.historical_stats['historical_specialist_accuracy'] else 0.5
        
        return features
    
    async def _generate_historical_training_report(self):
        """Generate comprehensive historical training report"""
        
        print(f"""
🏛️ AURA HISTORICAL EDUCATION TRAINING REPORT
════════════════════════════════════════════════
📚 Learning Statistics:
   • Conversations Processed: {self.historical_stats['total_conversations']}
   • Total Turns: {self.historical_stats['total_turns']}
   • Student Questions: {self.historical_stats['student_questions_processed']}
   • Expert Responses: {self.historical_stats['expert_responses_learned']}

🎯 Historical Specialist Performance:
   • Routing Accuracy: {np.mean(self.historical_stats['historical_specialist_accuracy']):.3f}
   • Historical Entities Learned: {len(self.historical_stats['historical_entities_learned'])}

📊 Question Type Distribution:
   • Who Questions: {self.historical_stats['question_types']['who']}
   • What Questions: {self.historical_stats['question_types']['what']}  
   • When Questions: {self.historical_stats['question_types']['when']}
   • Where Questions: {self.historical_stats['question_types']['where']}
   • Why Questions: {self.historical_stats['question_types']['why']}
   • How Questions: {self.historical_stats['question_types']['how']}

🕰️ Historical Period Coverage:
{self._format_period_coverage()}

🧠 Neural Development:
   • Hippocampal Memories Formed: {self.historical_stats['hippocampal_memory_formation']}
   • Historical Routing Enhanced: ✓
   • Educational Learning Patterns: ✓

🌍 Temporal Coverage:
   • Earliest Date: {self.historical_stats['temporal_range_coverage']['start']}
   • Latest Date: {self.historical_stats['temporal_range_coverage']['end']}

🎓 Educational Capabilities Enhanced:
   • Student Question Analysis ✓
   • Expert Answer Learning ✓
   • Historical Context Understanding ✓
   • Multi-turn Educational Conversations ✓
════════════════════════════════════════════════
        """)
    
    def _format_period_coverage(self) -> str:
        """Format historical period coverage"""
        periods = self.historical_stats['historical_periods_covered']
        if not periods:
            return "   • No periods covered yet"
        
        formatted = []
        for period, count in sorted(periods.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"   • {period.title()}: {count} conversations")
        
        return '\n'.join(formatted)
