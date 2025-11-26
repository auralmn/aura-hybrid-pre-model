#!/usr/bin/env python3
"""
Live System Health Monitor for AURA
Real-time HTML dashboard showing hormone levels, routing decisions, and system metrics
"""

import trio
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
from pathlib import Path

@dataclass
class SystemAlert:
    """System alert with severity and details"""
    timestamp: float
    severity: str  # 'critical', 'warning', 'info'
    component: str  # 'neuron', 'router', 'hormone', 'energy', etc.
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class FiringEvent:
    """Record of a neuron firing event"""
    timestamp: float
    neuron_id: str
    region: str
    firing_strength: float
    trigger_source: str  # 'input', 'attention', 'hormone', 'learning', etc.
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuronStatus:
    """Individual neuron status"""
    neuron_id: str
    firing_rate: float
    activity_state: str
    maturation_stage: str
    weight_magnitude: float
    last_fire_time: float
    is_healthy: bool
    firing_history: List[FiringEvent] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

@dataclass
class HealthSnapshot:
    """Single snapshot of system health"""
    timestamp: float
    hormone_levels: Dict[str, float]
    endocrine_effects: Dict[str, float]
    router_usage: Dict[str, int]
    expert_utilization: Dict[str, float]
    energy_consumption: float
    prediction_accuracy: float
    system_metrics: Dict[str, float]
    routing_decisions: List[Dict[str, Any]]
    neuron_statuses: List[NeuronStatus] = field(default_factory=list)
    active_alerts: List[SystemAlert] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class LiveHealthMonitor:
    """Real-time health monitoring with HTML dashboard"""
    
    def __init__(self, 
                 network,
                 update_interval: float = 1.0,
                 max_history: int = 1000,
                 html_output_path: str = "aura_health_dashboard.html",
                 enable_auto_adjustments: bool = True):
        self.network = network
        self.update_interval = update_interval
        self.max_history = max_history
        self.html_output_path = Path(html_output_path)
        self.enable_auto_adjustments = enable_auto_adjustments
        
        # History storage
        self.snapshots: deque = deque(maxlen=max_history)
        self.router_usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.hormone_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[trio.Task] = None
        self.total_routing_decisions = 0
        self.expert_collapse_warnings = 0
        
        # Alert system
        self.active_alerts: List[SystemAlert] = []
        self.alert_history: deque = deque(maxlen=500)
        self.alert_thresholds = {
            'neuron_firing_rate_min': 0.01,
            'neuron_weight_min': 1e-6,
            'hormone_level_min': 0.001,
            'hormone_level_max': 10.0,
            'energy_consumption_max': 1e-6,
            'prediction_accuracy_min': 0.1,
            'expert_utilization_min': 0.05,
            'router_confidence_min': 0.3
        }
        
        # Firing pattern recording
        self.firing_events: deque = deque(maxlen=10000)  # Last 10k firing events
        self.neuron_firing_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.region_firing_counts: Dict[str, int] = defaultdict(int)
        self.trigger_source_counts: Dict[str, int] = defaultdict(int)
        
        # Auto-adjustment thresholds
        self.dopamine_cap = 0.05
        self.temperature_base_increase = 0.1
        self.expert_collapse_threshold = 0.1  # If any expert usage < 10%
        
    async def start_monitoring(self):
        """Start the live monitoring system"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        print("🏥 Starting Live Health Monitor...")
        
        # Create initial HTML dashboard
        await self._generate_html_dashboard()
        
        print(f"✅ Health monitoring started - Dashboard: {self.html_output_path}")
        
        # Start monitoring loop in a nursery
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self._monitoring_loop)
            self.monitor_task = nursery
        
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
        print("🛑 Health monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._capture_snapshot()
                await self._check_expert_collapse()
                await self._generate_html_dashboard()
                await trio.sleep(self.update_interval)
            except trio.Cancelled:
                break
            except Exception as e:
                print(f"⚠️ Health monitor error: {e}")
                await trio.sleep(1.0)
                
    async def _capture_snapshot(self):
        """Capture current system state with comprehensive probing"""
        try:
            # Get hormone levels
            hormone_levels = {}
            if hasattr(self.network, '_pituitary') and self.network._pituitary:
                hormone_levels = self.network._pituitary.get_hormone_levels()
                # Convert enum keys to strings
                hormone_levels = {str(k): float(v) for k, v in hormone_levels.items()}
            
            # Get endocrine effects
            endocrine_effects = {}
            if hasattr(self.network, '_pituitary') and self.network._pituitary:
                effects = self.network._pituitary.apply_hormonal_effects(self.network)
                endocrine_effects = effects.get('effects', {})
            
            # Get router usage
            router_usage = {}
            if hasattr(self.network, '_thalamic_router') and self.network._thalamic_router:
                router_usage = dict(self.network._thalamic_router.routing_stats)
            
            # Get expert utilization
            expert_utilization = {}
            if hasattr(self.network, 'specialists'):
                for name, specialist in self.network.specialists.items():
                    if hasattr(specialist, 'nlms_head') and hasattr(specialist.nlms_head, 'w'):
                        weight_magnitude = np.mean(np.abs(specialist.nlms_head.w))
                        expert_utilization[name] = float(weight_magnitude)
            
            # Get energy consumption
            energy_consumption = 0.0
            if hasattr(self.network, '_thalamic_router') and hasattr(self.network._thalamic_router, 'moe'):
                moe = self.network._thalamic_router.moe
                if hasattr(moe, 'energy'):
                    energy_consumption = float(moe.energy.total_j)
            
            # Get prediction accuracy (proxy from last result)
            prediction_accuracy = getattr(self.network, '_last_acc', 0.5)
            
            # Get system metrics
            system_metrics = {}
            if hasattr(self.network, '_hypothalamus') and self.network._hypothalamus:
                system_metrics = self.network._hypothalamus.get_system_health()
            
            # Get recent routing decisions
            routing_decisions = []
            if hasattr(self.network, '_thalamic_router') and self.network._thalamic_router:
                router = self.network._thalamic_router
                if hasattr(router, 'routing_history'):
                    routing_decisions = list(router.routing_history)[-10:]  # Last 10 decisions
            
            # Probe all neurons
            neuron_statuses = await self._probe_all_neurons()
            
            # Check for alerts
            await self._check_system_alerts(hormone_levels, expert_utilization, energy_consumption, 
                                          prediction_accuracy, router_usage, neuron_statuses)
            
            # Create snapshot
            snapshot = HealthSnapshot(
                timestamp=time.time(),
                hormone_levels=hormone_levels,
                endocrine_effects=endocrine_effects,
                router_usage=router_usage,
                expert_utilization=expert_utilization,
                energy_consumption=energy_consumption,
                prediction_accuracy=prediction_accuracy,
                system_metrics=system_metrics,
                routing_decisions=routing_decisions,
                neuron_statuses=neuron_statuses,
                active_alerts=list(self.active_alerts)
            )
            
            self.snapshots.append(snapshot)
            
            # Update history
            for hormone, level in hormone_levels.items():
                self.hormone_history[hormone].append(level)
            for expert, usage in router_usage.items():
                self.router_usage_history[expert].append(usage)
                
        except Exception as e:
            print(f"⚠️ Error capturing snapshot: {e}")
            
    async def _probe_all_neurons(self) -> List[NeuronStatus]:
        """Probe all neurons in the system for health status"""
        neuron_statuses = []
        
        try:
            # Probe thalamus neurons
            if hasattr(self.network, '_thalamus') and hasattr(self.network._thalamus, 'neurons'):
                for i, neuron in enumerate(self.network._thalamus.neurons):
                    status = await self._probe_neuron(neuron, f"thalamus_{i}")
                    neuron_statuses.append(status)
            
            # Probe hippocampus neurons
            if hasattr(self.network, '_hippocampus') and hasattr(self.network._hippocampus, 'neurons'):
                for i, neuron in enumerate(self.network._hippocampus.neurons):
                    status = await self._probe_neuron(neuron, f"hippocampus_{i}")
                    neuron_statuses.append(status)
            
            # Probe amygdala neurons
            if hasattr(self.network, '_amygdala') and hasattr(self.network._amygdala, 'neurons'):
                for i, neuron in enumerate(self.network._amygdala.neurons):
                    status = await self._probe_neuron(neuron, f"amygdala_{i}")
                    neuron_statuses.append(status)
            
            # Probe router neurons
            if hasattr(self.network, '_thalamic_router') and hasattr(self.network._thalamic_router, 'routing_neurons'):
                router = self.network._thalamic_router
                for group_name, neurons in router.routing_neurons.items():
                    for i, neuron in enumerate(neurons):
                        status = await self._probe_neuron(neuron, f"router_{group_name}_{i}")
                        neuron_statuses.append(status)
                        
        except Exception as e:
            print(f"⚠️ Error probing neurons: {e}")
            
        return neuron_statuses
        
    async def _probe_neuron(self, neuron, neuron_id: str) -> NeuronStatus:
        """Probe individual neuron for health status and record firing events"""
        issues = []
        is_healthy = True
        firing_history = []
        
        try:
            # Get firing rate
            firing_rate = 0.0
            if hasattr(neuron, 'firing_rate'):
                firing_rate = float(neuron.firing_rate)
            elif hasattr(neuron, 'activity_state'):
                # Estimate firing rate from activity state
                if neuron.activity_state.value == 'active':
                    firing_rate = 0.5
                elif neuron.activity_state.value == 'inhibited':
                    firing_rate = 0.1
                else:
                    firing_rate = 0.0
            
            # Check if neuron is firing
            if firing_rate < self.alert_thresholds['neuron_firing_rate_min']:
                issues.append(f"Low firing rate: {firing_rate:.4f}")
                is_healthy = False
            
            # Get activity state
            activity_state = "unknown"
            if hasattr(neuron, 'activity_state'):
                activity_state = str(neuron.activity_state.value)
            
            # Get maturation stage
            maturation_stage = "unknown"
            if hasattr(neuron, 'maturation_stage'):
                maturation_stage = str(neuron.maturation_stage.value)
            
            # Get weight magnitude
            weight_magnitude = 0.0
            if hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'w'):
                weight_magnitude = float(np.mean(np.abs(neuron.nlms_head.w)))
            
            # Check weight health
            if weight_magnitude < self.alert_thresholds['neuron_weight_min']:
                issues.append(f"Low weight magnitude: {weight_magnitude:.2e}")
                is_healthy = False
            
            # Get last fire time
            last_fire_time = 0.0
            if hasattr(neuron, 'last_fire_time'):
                last_fire_time = float(neuron.last_fire_time)
            
            # Check if neuron hasn't fired recently
            current_time = time.time()
            if last_fire_time > 0 and (current_time - last_fire_time) > 60:  # 1 minute
                issues.append(f"Hasn't fired in {current_time - last_fire_time:.1f}s")
                is_healthy = False
            
            # Record firing event if neuron is active
            if firing_rate > 0.1:  # Threshold for recording firing events
                region = neuron_id.split('_')[0]  # Extract region from neuron_id
                
                # Determine trigger source based on context
                trigger_source = "unknown"
                trigger_details = {}
                context = {}
                
                # Check for attention-based firing
                if hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'spiking_attention'):
                    trigger_source = "attention"
                    trigger_details = {
                        "attention_gain": getattr(neuron.nlms_head.spiking_attention, 'gain_up', 1.0),
                        "attention_config": getattr(neuron.nlms_head.spiking_attention, 'config', {})
                    }
                
                # Check for hormone-based firing
                elif hasattr(neuron, 'hormone_sensitivity'):
                    trigger_source = "hormone"
                    trigger_details = {
                        "hormone_sensitivity": neuron.hormone_sensitivity
                    }
                
                # Check for learning-based firing
                elif hasattr(neuron, 'learning_rate') and neuron.learning_rate > 0:
                    trigger_source = "learning"
                    trigger_details = {
                        "learning_rate": neuron.learning_rate,
                        "weight_change": getattr(neuron, 'last_weight_change', 0.0)
                    }
                
                # Check for input-based firing
                elif hasattr(neuron, 'last_input_strength'):
                    trigger_source = "input"
                    trigger_details = {
                        "input_strength": neuron.last_input_strength
                    }
                
                # Add context information
                context = {
                    "activity_state": activity_state,
                    "maturation_stage": maturation_stage,
                    "weight_magnitude": weight_magnitude,
                    "region": region
                }
                
                # Record the firing event
                self.record_firing_event(
                    neuron_id=neuron_id,
                    region=region,
                    firing_strength=firing_rate,
                    trigger_source=trigger_source,
                    trigger_details=trigger_details,
                    context=context
                )
            
            # Get recent firing history for this neuron
            firing_history = self.get_neuron_firing_history(neuron_id, max_events=10)
            
            return NeuronStatus(
                neuron_id=neuron_id,
                firing_rate=firing_rate,
                activity_state=activity_state,
                maturation_stage=maturation_stage,
                weight_magnitude=weight_magnitude,
                last_fire_time=last_fire_time,
                is_healthy=is_healthy,
                firing_history=firing_history,
                issues=issues
            )
            
        except Exception as e:
            return NeuronStatus(
                neuron_id=neuron_id,
                firing_rate=0.0,
                activity_state="error",
                maturation_stage="error",
                weight_magnitude=0.0,
                last_fire_time=0.0,
                is_healthy=False,
                firing_history=[],
                issues=[f"Probe error: {str(e)}"]
            )
            
    async def _check_system_alerts(self, hormone_levels: Dict[str, float], 
                                 expert_utilization: Dict[str, float],
                                 energy_consumption: float,
                                 prediction_accuracy: float,
                                 router_usage: Dict[str, int],
                                 neuron_statuses: List[NeuronStatus]):
        """Check for system alerts and update active alerts"""
        
        # Clear resolved alerts
        self.active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        # Check hormone levels
        for hormone, level in hormone_levels.items():
            if level < self.alert_thresholds['hormone_level_min']:
                await self._add_alert('warning', 'hormone', 
                                    f"{hormone} level critically low: {level:.4f}",
                                    {'hormone': hormone, 'level': level})
            elif level > self.alert_thresholds['hormone_level_max']:
                await self._add_alert('critical', 'hormone',
                                    f"{hormone} level dangerously high: {level:.4f}",
                                    {'hormone': hormone, 'level': level})
        
        # Check expert utilization
        for expert, utilization in expert_utilization.items():
            if utilization < self.alert_thresholds['expert_utilization_min']:
                await self._add_alert('warning', 'expert',
                                    f"Expert {expert} underutilized: {utilization:.4f}",
                                    {'expert': expert, 'utilization': utilization})
        
        # Check energy consumption
        if energy_consumption > self.alert_thresholds['energy_consumption_max']:
            await self._add_alert('warning', 'energy',
                                f"High energy consumption: {energy_consumption:.2e} J",
                                {'energy': energy_consumption})
        
        # Check prediction accuracy
        if prediction_accuracy < self.alert_thresholds['prediction_accuracy_min']:
            await self._add_alert('critical', 'performance',
                                f"Low prediction accuracy: {prediction_accuracy:.3f}",
                                {'accuracy': prediction_accuracy})
        
        # Check router usage distribution
        total_usage = sum(router_usage.values())
        if total_usage > 0:
            for expert, usage in router_usage.items():
                usage_ratio = usage / total_usage
                if usage_ratio < self.expert_collapse_threshold:
                    await self._add_alert('warning', 'router',
                                        f"Expert {expert} usage collapse: {usage_ratio:.1%}",
                                        {'expert': expert, 'usage_ratio': usage_ratio})
        
        # Check neuron health
        unhealthy_neurons = [n for n in neuron_statuses if not n.is_healthy]
        if unhealthy_neurons:
            await self._add_alert('critical', 'neuron',
                                f"{len(unhealthy_neurons)} neurons unhealthy",
                                {'unhealthy_count': len(unhealthy_neurons), 
                                 'unhealthy_neurons': [n.neuron_id for n in unhealthy_neurons]})
        
        # Check for specific neuron issues
        for neuron in neuron_statuses:
            if not neuron.is_healthy:
                for issue in neuron.issues:
                    await self._add_alert('warning', 'neuron',
                                        f"Neuron {neuron.neuron_id}: {issue}",
                                        {'neuron_id': neuron.neuron_id, 'issue': issue})
        
    async def _add_alert(self, severity: str, component: str, message: str, details: Dict[str, Any]):
        """Add a new system alert"""
        # Check if similar alert already exists
        for alert in self.active_alerts:
            if (alert.severity == severity and 
                alert.component == component and 
                alert.message == message and 
                not alert.resolved):
                return  # Don't duplicate alerts
        
        alert = SystemAlert(
            timestamp=time.time(),
            severity=severity,
            component=component,
            message=message,
            details=details
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Print alert
        severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        print(f"{severity_emoji.get(severity, '⚪')} ALERT [{component.upper()}] {message}")
        
        # Auto-resolve old alerts
        current_time = time.time()
        self.active_alerts = [alert for alert in self.active_alerts 
                             if current_time - alert.timestamp < 300]  # 5 minutes
        
    def record_firing_event(self, neuron_id: str, region: str, firing_strength: float, 
                           trigger_source: str, trigger_details: Dict[str, Any] = None,
                           context: Dict[str, Any] = None):
        """Record a neuron firing event"""
        event = FiringEvent(
            timestamp=time.time(),
            neuron_id=neuron_id,
            region=region,
            firing_strength=firing_strength,
            trigger_source=trigger_source,
            trigger_details=trigger_details or {},
            context=context or {}
        )
        
        # Add to global firing events
        self.firing_events.append(event)
        
        # Add to neuron-specific history
        self.neuron_firing_history[neuron_id].append(event)
        
        # Update counters
        self.region_firing_counts[region] += 1
        self.trigger_source_counts[trigger_source] += 1
        
        # Print significant firing events
        if firing_strength > 0.8:  # High strength firing
            print(f"🔥 FIRING [{region.upper()}] {neuron_id} (strength: {firing_strength:.3f}, trigger: {trigger_source})")
        elif firing_strength > 0.5:  # Medium strength firing
            print(f"⚡ FIRE [{region.upper()}] {neuron_id} (strength: {firing_strength:.3f}, trigger: {trigger_source})")
            
    def get_firing_patterns(self, time_window: float = 60.0) -> Dict[str, Any]:
        """Get firing patterns within a time window"""
        current_time = time.time()
        recent_events = [e for e in self.firing_events if current_time - e.timestamp <= time_window]
        
        if not recent_events:
            return {"total_firings": 0, "patterns": {}}
        
        # Group by region
        region_patterns = defaultdict(list)
        for event in recent_events:
            region_patterns[event.region].append(event)
        
        # Group by trigger source
        trigger_patterns = defaultdict(list)
        for event in recent_events:
            trigger_patterns[event.trigger_source].append(event)
        
        # Calculate firing rates
        firing_rates = {}
        for region, events in region_patterns.items():
            firing_rates[region] = len(events) / time_window  # firings per second
        
        # Find most active neurons
        neuron_activity = defaultdict(int)
        for event in recent_events:
            neuron_activity[event.neuron_id] += 1
        
        most_active = sorted(neuron_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_firings": len(recent_events),
            "time_window": time_window,
            "firing_rates": dict(firing_rates),
            "region_patterns": {k: len(v) for k, v in region_patterns.items()},
            "trigger_patterns": {k: len(v) for k, v in trigger_patterns.items()},
            "most_active_neurons": most_active,
            "average_firing_strength": np.mean([e.firing_strength for e in recent_events]) if recent_events else 0.0
        }
        
    def get_neuron_firing_history(self, neuron_id: str, max_events: int = 50) -> List[FiringEvent]:
        """Get firing history for a specific neuron"""
        return list(self.neuron_firing_history[neuron_id])[-max_events:]
        
    def get_region_firing_summary(self, region: str) -> Dict[str, Any]:
        """Get firing summary for a specific region"""
        region_events = [e for e in self.firing_events if e.region == region]
        
        if not region_events:
            return {"total_firings": 0, "neurons": [], "triggers": {}}
        
        # Get unique neurons in this region
        neurons = list(set(e.neuron_id for e in region_events))
        
        # Get trigger sources
        triggers = defaultdict(int)
        for event in region_events:
            triggers[event.trigger_source] += 1
        
        # Calculate average firing strength
        avg_strength = np.mean([e.firing_strength for e in region_events])
        
        return {
            "total_firings": len(region_events),
            "neurons": neurons,
            "triggers": dict(triggers),
            "average_firing_strength": avg_strength,
            "most_active_neuron": max(neurons, key=lambda n: sum(1 for e in region_events if e.neuron_id == n))
        }
            
    async def _check_expert_collapse(self):
        """Check for expert collapse and apply auto-adjustments"""
        if not self.enable_auto_adjustments:
            return
            
        try:
            # Check if any expert has very low usage
            if not self.snapshots:
                return
                
            latest = self.snapshots[-1]
            total_usage = sum(latest.router_usage.values())
            
            if total_usage == 0:
                return
                
            # Check for expert collapse
            collapsed_experts = []
            for expert, usage in latest.router_usage.items():
                usage_ratio = usage / total_usage
                if usage_ratio < self.expert_collapse_threshold:
                    collapsed_experts.append(expert)
            
            if collapsed_experts:
                self.expert_collapse_warnings += 1
                print(f"⚠️ Expert collapse detected: {collapsed_experts}")
                
                # Apply auto-adjustments
                await self._apply_collapse_fixes()
                
        except Exception as e:
            print(f"⚠️ Error checking expert collapse: {e}")
            
    async def _apply_collapse_fixes(self):
        """Apply fixes for expert collapse"""
        try:
            # Cap dopamine nudge
            if hasattr(self.network, '_pituitary') and self.network._pituitary:
                # Reduce dopamine effect
                for hormone in self.network._pituitary.hormones.values():
                    if 'dopamine' in str(hormone.hormone_type).lower():
                        hormone.concentration = min(hormone.concentration, self.dopamine_cap)
            
            # Increase router temperature base
            if hasattr(self.network, '_thalamic_router') and self.network._thalamic_router:
                router = self.network._thalamic_router
                if hasattr(router, 'temperature'):
                    router.temperature += self.temperature_base_increase
                    print(f"🔧 Increased router temperature to {router.temperature}")
            
            print(f"🔧 Applied collapse fixes (dopamine cap: {self.dopamine_cap}, temp increase: {self.temperature_base_increase})")
            
        except Exception as e:
            print(f"⚠️ Error applying collapse fixes: {e}")
            
    async def _generate_html_dashboard(self):
        """Generate real-time HTML dashboard"""
        try:
            if not self.snapshots:
                return
                
            latest = self.snapshots[-1]
            
            # Prepare data for visualization
            chart_data = self._prepare_chart_data()
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURA Live Health Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(0,255,0,0.2);
            border: 1px solid #00ff00;
            margin-left: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .card h3 {{
            margin-top: 0;
            color: #ffd700;
            border-bottom: 2px solid #ffd700;
            padding-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            font-weight: bold;
        }}
        .metric-value {{
            color: #00ff00;
        }}
        .warning {{
            color: #ff6b6b;
            font-weight: bold;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 15px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            opacity: 0.7;
        }}
        .auto-refresh {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 10px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="auto-refresh">
        🔄 Auto-refresh: {self.update_interval}s<br>
        📊 Snapshots: {len(self.snapshots)}<br>
        ⚠️ Collapse warnings: {self.expert_collapse_warnings}
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🧠 AURA Live Health Monitor</h1>
            <div class="status">🟢 ACTIVE</div>
        </div>
        
        <div class="grid">
            <!-- Hormone Levels -->
            <div class="card">
                <h3>🧬 Hormone Levels</h3>
                {self._format_hormone_metrics(latest.hormone_levels)}
            </div>
            
            <!-- Endocrine Effects -->
            <div class="card">
                <h3>⚡ Endocrine Effects</h3>
                {self._format_effects_metrics(latest.endocrine_effects)}
            </div>
            
            <!-- Router Usage -->
            <div class="card">
                <h3>🎯 Router Usage</h3>
                {self._format_router_metrics(latest.router_usage)}
                <div class="chart-container">
                    <canvas id="routerChart"></canvas>
                </div>
            </div>
            
            <!-- Expert Utilization -->
            <div class="card">
                <h3>🔬 Expert Utilization</h3>
                {self._format_expert_metrics(latest.expert_utilization)}
                <div class="chart-container">
                    <canvas id="expertChart"></canvas>
                </div>
            </div>
            
            <!-- System Metrics -->
            <div class="card">
                <h3>📊 System Metrics</h3>
                {self._format_system_metrics(latest.system_metrics)}
            </div>
            
            <!-- Energy & Performance -->
            <div class="card">
                <h3>⚡ Energy & Performance</h3>
                <div class="metric">
                    <span class="metric-label">Energy Consumption:</span>
                    <span class="metric-value">{latest.energy_consumption:.2e} J</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Prediction Accuracy:</span>
                    <span class="metric-value">{latest.prediction_accuracy:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Routing Decisions:</span>
                    <span class="metric-value">{self.total_routing_decisions}</span>
                </div>
            </div>
        </div>
        
            <!-- Firing Patterns -->
            <div class="card">
                <h3>🔥 Firing Patterns</h3>
                {_format_firing_patterns(self)}
            </div>
            
            <!-- Active Alerts -->
            <div class="card">
                <h3>🚨 Active Alerts</h3>
                {_format_active_alerts(latest.active_alerts)}
            </div>
            
            <!-- Recent Routing Decisions -->
            <div class="card">
                <h3>🔄 Recent Routing Decisions</h3>
                {self._format_routing_decisions(latest.routing_decisions)}
            </div>
        
        <div class="footer">
            <p>🕒 Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest.timestamp))}</p>
            <p>🔄 Auto-refresh every {self.update_interval} seconds</p>
        </div>
    </div>
    
    <script>
        // Router Usage Chart
        const routerCtx = document.getElementById('routerChart').getContext('2d');
        new Chart(routerCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(latest.router_usage.keys()))},
                datasets: [{{
                    data: {json.dumps(list(latest.router_usage.values()))},
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: 'white' }}
                    }}
                }}
            }}
        }});
        
        // Expert Utilization Chart
        const expertCtx = document.getElementById('expertChart').getContext('2d');
        new Chart(expertCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(list(latest.expert_utilization.keys()))},
                datasets: [{{
                    label: 'Utilization',
                    data: {json.dumps(list(latest.expert_utilization.values()))},
                    backgroundColor: 'rgba(0, 255, 0, 0.6)',
                    borderColor: 'rgba(0, 255, 0, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{ color: 'white' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    x: {{
                        ticks: {{ color: 'white' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: 'white' }}
                    }}
                }}
            }}
        }});
        
        // Auto-refresh every {self.update_interval} seconds
        setTimeout(() => {{
            location.reload();
        }}, {int(self.update_interval * 1000)});
    </script>
</body>
</html>
"""
            
            # Write HTML file
            self.html_output_path.write_text(html_content, encoding='utf-8')
            
        except Exception as e:
            print(f"⚠️ Error generating HTML dashboard: {e}")
            
    def _format_hormone_metrics(self, hormones: Dict[str, float]) -> str:
        """Format hormone metrics for HTML"""
        if not hormones:
            return "<div class='metric'><span class='metric-label'>No data</span></div>"
            
        html = ""
        for hormone, level in hormones.items():
            color = "metric-value"
            if level > 1.5:
                color = "warning"
            html += f"""
            <div class="metric">
                <span class="metric-label">{hormone.replace('_', ' ').title()}:</span>
                <span class="{color}">{level:.3f}</span>
            </div>
            """
        return html
        
    def _format_effects_metrics(self, effects: Dict[str, float]) -> str:
        """Format endocrine effects for HTML"""
        if not effects:
            return "<div class='metric'><span class='metric-label'>No effects</span></div>"
            
        html = ""
        for effect, value in effects.items():
            color = "metric-value"
            if abs(value - 1.0) > 0.5:
                color = "warning"
            html += f"""
            <div class="metric">
                <span class="metric-label">{effect.replace('_', ' ').title()}:</span>
                <span class="{color}">{value:.3f}</span>
            </div>
            """
        return html
        
    def _format_router_metrics(self, usage: Dict[str, int]) -> str:
        """Format router usage for HTML"""
        if not usage:
            return "<div class='metric'><span class='metric-label'>No usage data</span></div>"
            
        total = sum(usage.values())
        html = ""
        for expert, count in sorted(usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            color = "metric-value"
            if percentage < 10:
                color = "warning"
            html += f"""
            <div class="metric">
                <span class="metric-label">{expert.replace('_', ' ').title()}:</span>
                <span class="{color}">{count} ({percentage:.1f}%)</span>
            </div>
            """
        return html
        
    def _format_expert_metrics(self, utilization: Dict[str, float]) -> str:
        """Format expert utilization for HTML"""
        if not utilization:
            return "<div class='metric'><span class='metric-label'>No utilization data</span></div>"
            
        html = ""
        for expert, util in sorted(utilization.items(), key=lambda x: x[1], reverse=True):
            color = "metric-value"
            if util < 0.1:
                color = "warning"
            html += f"""
            <div class="metric">
                <span class="metric-label">{expert.replace('_', ' ').title()}:</span>
                <span class="{color}">{util:.3f}</span>
            </div>
            """
        return html
        
    def _format_system_metrics(self, metrics: Dict[str, float]) -> str:
        """Format system metrics for HTML"""
        if not metrics:
            return "<div class='metric'><span class='metric-label'>No system data</span></div>"
            
        html = ""
        for metric, value in metrics.items():
            color = "metric-value"
            if 'stress' in metric.lower() and value > 0.7:
                color = "warning"
            html += f"""
            <div class="metric">
                <span class="metric-label">{metric.replace('_', ' ').title()}:</span>
                <span class="{color}">{value:.3f}</span>
            </div>
            """
        return html
        
    def _format_routing_decisions(self, decisions: List[Dict[str, Any]]) -> str:
        """Format recent routing decisions for HTML"""
        if not decisions:
            return "<div class='metric'><span class='metric-label'>No recent decisions</span></div>"
            
        html = ""
        for i, decision in enumerate(decisions[-5:]):  # Last 5 decisions
            query = decision.get('query', 'N/A')[:50] + "..." if len(decision.get('query', '')) > 50 else decision.get('query', 'N/A')
            primary = decision.get('routing_decision', {}).get('primary_target', 'N/A')
            confidence = decision.get('routing_decision', {}).get('routing_confidence', 0.0)
            
            html += f"""
            <div class="metric">
                <span class="metric-label">Decision {i+1}:</span>
                <span class="metric-value">{primary} ({confidence:.2f}) - "{query}"</span>
            </div>
            """
        return html
        
    def _prepare_chart_data(self) -> Dict[str, Any]:
        """Prepare data for charts"""
        if not self.snapshots:
            return {}
            
        # Prepare time series data
        timestamps = [s.timestamp for s in self.snapshots]
        
        # Hormone levels over time
        hormone_data = {}
        for hormone in self.hormone_history:
            hormone_data[hormone] = list(self.hormone_history[hormone])
            
        return {
            'timestamps': timestamps,
            'hormone_data': hormone_data,
            'router_usage': dict(self.router_usage_history)
        }

# Integration with Network
async def start_health_monitoring(network, 
                                update_interval: float = 1.0,
                                html_output_path: str = "aura_health_dashboard.html",
                                enable_auto_adjustments: bool = True):
    """Start health monitoring for a network"""
    monitor = LiveHealthMonitor(
        network=network,
        update_interval=update_interval,
        html_output_path=html_output_path,
        enable_auto_adjustments=enable_auto_adjustments
    )
    
    await monitor.start_monitoring()
    return monitor

def _format_firing_patterns(monitor) -> str:
    """Format firing patterns for HTML"""
    patterns = monitor.get_firing_patterns(time_window=60.0)
    
    if patterns["total_firings"] == 0:
        return "<div class='metric'><span class='metric-label'>No recent firing activity</span></div>"
    
    html = f"""
    <div class="metric">
        <span class="metric-label">Total Firings (60s):</span>
        <span class="metric-value">{patterns['total_firings']}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Average Firing Strength:</span>
        <span class="metric-value">{patterns['average_firing_strength']:.3f}</span>
    </div>
    """
    
    # Firing rates by region
    if patterns['firing_rates']:
        html += "<div class='metric'><span class='metric-label'>Firing Rates (Hz):</span></div>"
        for region, rate in patterns['firing_rates'].items():
            color = "metric-value"
            if rate > 10:
                color = "warning"
            html += f"""
            <div class="metric" style="margin-left: 20px;">
                <span class="metric-label">{region}:</span>
                <span class="{color}">{rate:.2f}</span>
            </div>
            """
    
    # Most active neurons
    if patterns['most_active_neurons']:
        html += "<div class='metric'><span class='metric-label'>Most Active Neurons:</span></div>"
        for neuron_id, count in patterns['most_active_neurons'][:5]:
            html += f"""
            <div class="metric" style="margin-left: 20px;">
                <span class="metric-label">{neuron_id}:</span>
                <span class="metric-value">{count} fires</span>
            </div>
            """
    
    return html

def _format_active_alerts(alerts: List[SystemAlert]) -> str:
    """Format active alerts for HTML"""
    if not alerts:
        return "<div class='metric'><span class='metric-label' style='color: #00ff00;'>✅ No active alerts</span></div>"
    
    html = ""
    for alert in alerts[-10:]:  # Last 10 alerts
        severity_color = {
            'critical': '#ff6b6b',
            'warning': '#ffd93d', 
            'info': '#6bcf7f'
        }.get(alert.severity, '#ffffff')
        
        time_ago = time.time() - alert.timestamp
        time_str = f"{time_ago:.0f}s ago" if time_ago < 60 else f"{time_ago/60:.0f}m ago"
        
        html += f"""
        <div class="metric" style="border-left: 4px solid {severity_color}; padding-left: 10px;">
            <span class="metric-label" style="color: {severity_color};">
                [{alert.severity.upper()}] {alert.component.upper()}
            </span>
            <span class="metric-value" style="color: {severity_color};">
                {time_str}
            </span>
        </div>
        <div class="metric" style="margin-left: 20px; font-size: 0.9em; color: #cccccc;">
            {alert.message}
        </div>
        """
    
    return html

print("🏥 AURA Live Health Monitor Loaded!")
print("=" * 50)
print("Features:")
print("• Real-time HTML dashboard")
print("• Hormone level monitoring")
print("• Router usage histograms")
print("• Expert collapse detection")
print("• Auto-adjustment capabilities")
print("• Energy consumption tracking")
print("• Performance metrics")
