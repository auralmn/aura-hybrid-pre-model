import numpy as np
from typing import Dict, List, Any

from .neuron import Neuron, ActivityState, MaturationStage
from ..relays.thalamic_router_relay import ThalamicRouterModule
from .attention import MultiChannelSpikingAttention, prosody_channels_from_text, RouterAttentionPresets
from .attention_telemetry import AttentionTelemetryBuffer, AttentionEvent, AttentionTelemetryLogger
from .liquid_moe import LiquidMoERouter, NLMSExpertAdapter, create_liquid_moe_from_router, attention_gain_from_text
import time



class ThalamicConversationRouter:
    def __init__(self, neuron_count: int = 60, features: int = 384, input_dim: int = 384, routing_confidence_threshold: float = 0.6, 
                 enable_attention: bool = False):
        self.confidence_threshold = float(routing_confidence_threshold)
        
        # Initialize attention system if enabled
        self.attn: MultiChannelSpikingAttention | None = None
        self.attn_buf = AttentionTelemetryBuffer(maxlen=2000)
        self.attn_hook = None  # Optional[Callable[[AttentionEvent], None]]
        if enable_attention:
            # Use conversational preset as default
            self.attn = RouterAttentionPresets.conversational()
        
        # Initialize Liquid-MoE router (will be set up after neuron groups are created)
        self.moe: LiquidMoERouter | None = None
        per_group = max(1, neuron_count // 6)
        def mk_group(prefix: str) -> List[Neuron]:
            return [
                Neuron(
                    neuron_id=f'{prefix}_{i}',
                    specialization=f'{prefix}_router',
                    abilities={'routing': 0.9},
                    maturation=MaturationStage.DIFFERENTIATED,
                    activity=ActivityState.RESTING,
                    n_features=features,
                    n_outputs=1,
                ) for i in range(per_group)
            ]

        self.routing_neurons: Dict[str, List[Neuron]] = {
            'general_chat': mk_group('general_chat'),
            'historical_specialist': mk_group('historical'),
            'amygdala_specialist': mk_group('amygdala'),
            'hippocampus_specialist': mk_group('memory'),
            'analytical_specialist': mk_group('analysis'),
            'multi_specialist': mk_group('multi'),
        }

        
        for group in self.routing_neurons.values():
            for n in group:
                n.nlms_head.clamp = (0.0, 1.0)
        self.all_neurons: List[Neuron] = [n for g in self.routing_neurons.values() for n in g]
        self.routing_relay = ThalamicRouterModule(input_dim=input_dim, routing_threshold=self.confidence_threshold)
        self.routing_history: List[Dict[str, Any]] = []
        self.routing_stats: Dict[str, int] = {k: 0 for k in self.routing_neurons.keys()}
        
        # Initialize Liquid-MoE router after neuron groups are created
        self.moe = create_liquid_moe_from_router(
            self, features=input_dim, hidden_dim=64, top_k=2, temperature=1.0
        )
        
        # Note: Individual neuron attention is now handled by the router-level attention system
        # The new multi-channel attention system provides μ scaling without touching NLMS internals
    

    def _has_historical_entities(self, q: str) -> bool:
        ents = ['egypt','rome','china','greece','napoleon','caesar','alexander','mongol','viking','crusade','renaissance','industrial revolution','world war','pharaoh','emperor','dynasty']
        return any(e in q for e in ents)
    
    def _attention_gain_from_text(self, text: str) -> float:
        """Extract attention gain from text using existing attention system"""
        return attention_gain_from_text(text, self.attn)
    
    async def adaptive_routing_update_with_attention(self, routing_plan: Dict[str, Any], 
                                               conversation_outcome: Dict[str, Any], 
                                               query_features: np.ndarray,
                                               query_text: str = "") -> None:
        """Enhanced routing update with Liquid-MoE and multi-channel spike attention."""
        success_score = float(conversation_outcome.get('user_satisfaction', 0.5))
        response_quality = float(conversation_outcome.get('response_quality', 0.5))
        routing_success = 0.5 * (success_score + response_quality)

        # Get attention gain for MoE routing
        attn_gain = self._attention_gain_from_text(query_text)
        y = 1.0 if routing_success > 0.7 else 0.0

        # Use Liquid-MoE router for learning
        moe_out = await self.moe.learn(query_features, y_true=y, attn_gain=attn_gain)

        # Prepare telemetry data
        mu_boost = attn_gain
        winners_idx = []
        sal_mean = sal_max = sal_std = 0.0
        spike_rate_amp = spike_rate_pitch = spike_rate_boundary = 0.0

        if self.attn is not None and query_text:
            tokens = query_text.split()
            token_ids = [hash(t.lower()) % 50000 for t in tokens]
            amp, pitch, boundary = prosody_channels_from_text(tokens)
            attn_res = self.attn.compute(token_ids, amp, pitch, boundary)
            winners_idx = list(attn_res["winners_idx"])

            sal = np.asarray(attn_res["salience"], dtype=np.float64)
            sal_mean = float(sal.mean()) if sal.size else 0.0
            sal_max = float(sal.max()) if sal.size else 0.0
            sal_std = float(sal.std()) if sal.size else 0.0

            spikes = attn_res["spikes"]
            T = max(1, len(tokens))
            spike_rate_amp = float(np.sum(spikes["amp"])) / T
            spike_rate_pitch = float(np.sum(spikes["pitch"])) / T
            spike_rate_boundary = float(np.sum(spikes["boundary"])) / T

        # Extract primary from MoE output
        primary = max(moe_out['per_expert'].keys(), key=lambda k: moe_out['per_expert'][k]['gate'])

        # Push telemetry event
        ev = AttentionEvent(
            t=time.time(),
            text_len=len(query_text.split()),
            mu_scalar=mu_boost,
            mu_applied_ratio=float(mu_boost),
            winners_idx=winners_idx,
            salience_mean=sal_mean,
            salience_max=sal_max,
            salience_std=sal_std,
            spike_rate_amp=spike_rate_amp,
            spike_rate_pitch=spike_rate_pitch,
            spike_rate_boundary=spike_rate_boundary,
            primary=primary,
            routing_confidence=float(routing_plan.get('confidence', 0.0)),
            routing_success=float(routing_success),
            note="adaptive_routing_update_with_attention",
        )
        self.attn_buf.push(ev)
        if self.attn_hook:
            try:
                self.attn_hook(ev)
            except Exception:
                pass

    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        ql = query.lower()
        char = {
            'is_greeting': any(w in ql for w in ['hello','hi','hey','good morning','good evening']),
            'is_question': ('?' in query) or any(w in ql for w in ['what','why','how','when','where','who']),
            'is_historical': any(w in ql for w in ['history','historical','ancient','medieval','war','empire','dynasty']) or self._has_historical_entities(ql),
            'is_emotional': any(w in ql for w in ['feel','scared','worried','excited','angry']),
            'is_memory_query': any(w in ql for w in ['remember','recall','told me','mentioned','previous']),
            'is_analytical': any(w in ql for w in ['analyze','compare','explain','because','causes','effects']),
            'is_casual': any(w in ql for w in ['chat','talk','discuss','just curious']),
            'has_multiple_questions': query.count('?') > 1,
            'is_compound_query': any(w in ql for w in [' and ',' also ',' furthermore ',' additionally ']),
            'word_count': len(query.split()),
        }
        factors = [
            char['word_count'] > 20,
            char['has_multiple_questions'],
            char['is_compound_query'],
            char['is_analytical'],
        ]
        char['complexity_score'] = sum(factors) / len(factors) if factors else 0.0
        return char

    def analyze_conversation_intent(self, user_query: str, query_features: np.ndarray) -> Dict[str, Any]:
        # Heuristic meta from query
        meta = self._analyze_query_characteristics(user_query)
        
        # Compute attention gain if available
        attn_gain = self._attention_gain_from_text(user_query)
        ev_stub = None
        
        # Prepare telemetry data if attention is enabled
        if self.attn is not None:
            tokens = user_query.split()
            if tokens:
                amp, pitch, boundary = prosody_channels_from_text(tokens)
                token_ids = [hash(t.lower()) % 50000 for t in tokens]
                ar = self.attn.compute(token_ids, amp, pitch, boundary)
                
                # Prepare telemetry data
                sal = np.asarray(ar["salience"], dtype=np.float64)
                ev_stub = {
                    "mu_scalar": attn_gain,
                    "winners_idx": list(ar["winners_idx"]),
                    "salience_mean": float(sal.mean()) if sal.size else 0.0,
                    "salience_max": float(sal.max()) if sal.size else 0.0,
                    "salience_std": float(sal.std()) if sal.size else 0.0,
                    "spike_rate_amp": float(np.sum(ar["spikes"]["amp"])) / max(1, len(tokens)),
                    "spike_rate_pitch": float(np.sum(ar["spikes"]["pitch"])) / max(1, len(tokens)),
                    "spike_rate_boundary": float(np.sum(ar["spikes"]["boundary"])) / max(1, len(tokens)),
                    "text_len": len(tokens),
                }
        
        # Use Liquid-MoE router for expert selection
        moe_out = self.moe.route(query_features, attn_gain=attn_gain)
        
        # Extract primary target and confidence from MoE
        primary = max(moe_out['per_expert'].keys(), key=lambda k: moe_out['per_expert'][k]['gate'])
        conf = float(max(v['gate'] for v in moe_out['per_expert'].values()))
        
        # Build routing scores from MoE output
        routing_scores: Dict[str, Dict[str, Any]] = {}
        for name, info in moe_out['per_expert'].items():
            routing_scores[name] = {
                'confidence': info['gate'],
                'consistency': 1.0,  # MoE provides single prediction per expert
                'activation_pattern': [info['pred']]
            }
        
        # Blend with heuristic for additional context
        for spec in routing_scores:
            heuristic_score = self.routing_relay.score(meta)
            routing_scores[spec]['confidence'] = 0.7 * routing_scores[spec]['confidence'] + 0.3 * heuristic_score
        primary = max(routing_scores.keys(), key=lambda k: routing_scores[k]['confidence'])
        high = [k for k, v in routing_scores.items() if v['confidence'] > self.confidence_threshold]
        strategy = self._determine_routing_strategy(routing_scores, meta)
        # Log telemetry event for intent analysis
        if ev_stub is not None:
            ev = AttentionEvent(
                t=time.time(),
                text_len=ev_stub["text_len"],
                mu_scalar=ev_stub["mu_scalar"],
                mu_applied_ratio=1.0,  # no μ update here, just inference gain usage
                winners_idx=ev_stub["winners_idx"],
                salience_mean=ev_stub["salience_mean"],
                salience_max=ev_stub["salience_max"],
                salience_std=ev_stub["salience_std"],
                spike_rate_amp=ev_stub["spike_rate_amp"],
                spike_rate_pitch=ev_stub["spike_rate_pitch"],
                spike_rate_boundary=ev_stub["spike_rate_boundary"],
                primary=None,
                routing_confidence=None,
                routing_success=None,
                note="intent_analysis_attention",
            )
            self.attn_buf.push(ev)
            if self.attn_hook:
                try:
                    self.attn_hook(ev)
                except Exception:
                    pass

        return {
            'primary_target': primary,
            'routing_confidence': routing_scores[primary]['confidence'],
            'needs_multiple_specialists': len(high) > 1,
            'secondary_targets': high,
            'routing_scores': routing_scores,
            'query_characteristics': meta,
            'routing_strategy': strategy,
            'attention_gain': attn_gain,
            'moe': moe_out,  # Include full MoE debug info
        }

    def get_attention_telemetry(self, query_text: str) -> Dict[str, Any]:
        """Get attention telemetry for logging and monitoring"""
        if self.attn is None or not query_text:
            return {'enabled': False}
        
        tokens = query_text.split()
        token_ids = [hash(t.lower()) % 50000 for t in tokens]
        amp, pitch, boundary = prosody_channels_from_text(tokens)
        attn_res = self.attn.compute(token_ids, amp, pitch, boundary)
        
        return {
            'enabled': True,
            'mu_scalar': attn_res['mu_scalar'],
            'winners_count': len(attn_res['winners_idx']),
            'winners_idx': attn_res['winners_idx'],
            'avg_salience': float(np.mean(attn_res['salience'])) if len(attn_res['salience']) > 0 else 0.0,
            'max_salience': float(np.max(attn_res['salience'])) if len(attn_res['salience']) > 0 else 0.0,
            'spike_counts': {
                'amplitude': int(np.sum(attn_res['spikes']['amp'])),
                'pitch': int(np.sum(attn_res['spikes']['pitch'])),
                'boundary': int(np.sum(attn_res['spikes']['boundary']))
            },
            'tokens': tokens,
            'prosody_channels': {
                'amplitude': amp.tolist(),
                'pitch': pitch.tolist(),
                'boundary': boundary.tolist()
            }
        }

    def get_attention_summary(self) -> Dict[str, float]:
        """Get attention telemetry summary statistics"""
        return self.attn_buf.summary()

    def recent_attention_events(self, n: int = 20) -> List[Dict]:
        """Get recent attention events as dictionaries"""
        return self.attn_buf.recent_dicts(n)

    def set_attention_hook(self, fn) -> None:
        """Set a callback function for attention events"""
        self.attn_hook = fn

    def clear_attention_telemetry(self) -> None:
        """Clear all attention telemetry data"""
        self.attn_buf.clear()

    def get_moe_stats(self) -> Dict[str, Any]:
        """Get Liquid-MoE router statistics"""
        if self.moe is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'usage_stats': self.moe.get_usage_stats(),
            'energy_stats': self.moe.get_energy_stats(),
            'n_experts': len(self.moe.names),
            'top_k': self.moe.top_k,
            'temperature': self.moe.temperature
        }

    def reset_moe(self) -> None:
        """Reset Liquid-MoE router state"""
        if self.moe is not None:
            self.moe.reset()

    def get_moe_usage_balance(self) -> Dict[str, float]:
        """Get expert usage balance for load monitoring"""
        if self.moe is None:
            return {}
        
        usage_stats = self.moe.get_usage_stats()
        return {
            'usage_ma': dict(zip(self.moe.names, usage_stats['usage_ma'])),
            'target_usage': usage_stats['target_usage'],
            'usage_std': usage_stats['usage_std'],
            'usage_entropy': usage_stats['usage_entropy']
        }

    def _determine_routing_strategy(self, routing_scores: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> str:
        max_conf = max(v['confidence'] for v in routing_scores.values())
        high_count = sum(1 for v in routing_scores.values() if v['confidence'] > self.confidence_threshold)
        if meta['is_greeting'] or meta['is_casual']:
            return 'direct_to_general'
        if max_conf > 0.9:
            return 'direct_routing'
        if high_count > 1:
            return 'parallel_processing'
        if meta['complexity_score'] > 0.6:
            return 'staged_routing'
        return 'default_routing'

    def route_conversation(self, routing_decision: Dict[str, Any], user_query: str, query_features: np.ndarray) -> Dict[str, Any]:
        strategy = routing_decision['routing_strategy']
        primary = routing_decision['primary_target']
        plan: Dict[str, Any] = {
            'strategy': strategy,
            'primary_specialist': primary,
            'routing_sequence': [],
            'parallel_processing': False,
            'confidence': routing_decision['routing_confidence'],
        }
        if strategy == 'direct_to_general':
            plan['routing_sequence'] = ['general_chat']
        elif strategy == 'direct_routing':
            plan['routing_sequence'] = [primary]
        elif strategy == 'parallel_processing':
            plan['parallel_processing'] = True
            plan['routing_sequence'] = routing_decision['secondary_targets']
        elif strategy == 'staged_routing':
            seq = [primary]
            char = routing_decision['query_characteristics']
            if char['is_historical'] and primary != 'historical_specialist':
                seq.append('historical_specialist')
            if char['is_emotional'] and primary != 'amygdala_specialist':
                seq.append('amygdala_specialist')
            if char['is_memory_query'] and primary != 'hippocampus_specialist':
                seq.append('hippocampus_specialist')
            plan['routing_sequence'] = seq
        else:
            plan['routing_sequence'] = [primary, 'general_chat']
        self.routing_stats[primary] = self.routing_stats.get(primary, 0) + 1
        self.routing_history.append({'query': user_query, 'routing_decision': routing_decision, 'routing_plan': plan, 'timestamp': len(self.routing_history)})
        
        # Remember winners for dopamine reward
        self._remember_winners([primary] + routing_decision.get('secondary_targets', []))
        
        return plan

    def _remember_winners(self, idxs):
        """Remember winners for dopamine reward"""
        try:
            import numpy as np
            self.last_winners = np.array(list(map(int, idxs)), dtype=np.int64)
        except Exception:
            self.last_winners = None

    async def adaptive_routing_update(self, routing_plan: Dict[str, Any], conversation_outcome: Dict[str, Any], query_features: np.ndarray) -> None:
        success_score = float(conversation_outcome.get('user_satisfaction', 0.5))
        response_quality = float(conversation_outcome.get('response_quality', 0.5))
        routing_success = 0.5 * (success_score + response_quality)
        primary = routing_plan.get('primary_specialist')
        targets = self.routing_neurons.get(primary, []) # type: ignore
        y = 1.0 if routing_success > 0.7 else 0.0
        for n in targets:
            await n.update_nlms(query_features, y)

    def get_routing_statistics(self) -> Dict[str, Any]:
        total = sum(self.routing_stats.values())
        if total == 0:
            return {'message': 'No routing decisions yet'}
        dist = {k: v / total for k, v in self.routing_stats.items()}
        most_used = max(self.routing_stats.keys(), key=lambda k: self.routing_stats[k])
        avg_conf = float(np.mean([r['routing_decision']['routing_confidence'] for r in self.routing_history])) if self.routing_history else 0.0
        eff = len([r for r in self.routing_history if len(r['routing_plan']['routing_sequence']) == 1]) / total
        return {'total_conversations': total, 'routing_distribution': dist, 'most_used_specialist': most_used, 'average_routing_confidence': avg_conf, 'routing_efficiency': eff}

    def explain_routing_decision(self, routing_decision: Dict[str, Any]) -> str:
        primary = routing_decision.get('primary_target')
        conf = routing_decision.get('routing_confidence', 0.0)
        strategy = routing_decision.get('routing_strategy', 'default')
        meta = routing_decision.get('query_characteristics', {})
        bits = []
        if meta.get('is_historical'): bits.append('historical cues')
        if meta.get('is_emotional'): bits.append('emotional words')
        if meta.get('is_memory_query'): bits.append('memory/recall')
        if meta.get('is_analytical'): bits.append('analytical intent')
        if not bits: bits.append('general chat indicators')
        return f"Routed to {primary} (conf={conf:.2f}) via {strategy} due to {', '.join(bits)}."

    async def process(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process input through the thalamic router with Qdrant streaming"""
        # Analyze the input to determine routing characteristics
        query_text = ""  # We don't have text in this context, but we can still analyze features
        characteristics = self._analyze_query_characteristics(query_text)
        
        # Get routing decision
        routing_decision = self.analyze_conversation_intent(query_text, input_data)
        
        # Route the conversation
        routing_result = self.route_conversation(routing_decision, query_text, input_data)
        
        return {
            'routing_decision': routing_decision,
            'routing_result': routing_result,
            'characteristics': characteristics
        }

    async def init_weights(self):
        """Initialize weights for all neurons in the thalamic router"""
        for neuron in self.all_neurons:
            if hasattr(neuron, 'init_weights'):
                await neuron.init_weights()
            elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                await neuron.nlms_head.init_weights()

