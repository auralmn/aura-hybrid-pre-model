#!/usr/bin/env python3
"""
AURA WebSocket Bridge
Connects AURA's health monitoring system to the Next.js dashboard via WebSocket
"""

import asyncio
import json
import time
import trio
import socketio
from typing import Dict, Any, Optional
import numpy as np

from .health_monitor import LiveHealthMonitor
from ..core.network import Network


class AURAWebSocketBridge:
    """Bridge between AURA health monitoring and WebSocket dashboard"""
    
    def __init__(self, server_url: str = "http://localhost:3001"):
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.network: Optional[Network] = None
        self.health_monitor: Optional[LiveHealthMonitor] = None
        self.is_connected = False
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.event
        async def connect():
            print("🔌 Connected to WebSocket server")
            self.is_connected = True
            await self.sio.emit('aura_connected', {
                'status': 'connected',
                'timestamp': time.time()
            })
        
        @self.sio.event
        async def disconnect():
            print("🔌 Disconnected from WebSocket server")
            self.is_connected = False
        
        @self.sio.event
        async def request_health_snapshot():
            """Handle request for current health snapshot"""
            if self.health_monitor:
                snapshot = await self.health_monitor._capture_snapshot()
                await self.sio.emit('health_snapshot', snapshot)
        
        @self.sio.event
        async def request_firing_patterns(data):
            """Handle request for firing patterns"""
            if self.health_monitor:
                time_window = data.get('time_window', 60.0)
                patterns = self.health_monitor.get_firing_patterns(time_window)
                await self.sio.emit('firing_patterns', patterns)
    
    async def connect_to_server(self):
        """Connect to the WebSocket server"""
        try:
            await self.sio.connect(self.server_url)
            return True
        except Exception as e:
            print(f"❌ Failed to connect to WebSocket server: {e}")
            return False
    
    async def initialize_aura_network(self):
        """Initialize AURA network with health monitoring"""
        try:
            print("🧠 Initializing AURA network...")
            
            # Create network with health monitoring
            self.network = Network(
                enable_qdrant_streaming=True,
                enable_endocrine=True,
                enable_hippocampus_bias=True
            )
            
            # Initialize network
            await self.network.initialize()
            
            # Create health monitor
            self.health_monitor = LiveHealthMonitor(
                network=self.network,
                update_interval=2.0,
                html_output_path="aura_health_dashboard.html",
                enable_auto_adjustments=True
            )
            
            print("✅ AURA network initialized with health monitoring")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize AURA network: {e}")
            return False
    
    async def start_health_monitoring(self):
        """Start the health monitoring loop"""
        if not self.health_monitor:
            print("❌ Health monitor not initialized")
            return
        
        print("🏥 Starting health monitoring...")
        
        # Start monitoring in background
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self._monitoring_loop)
            nursery.start_soon(self._simulation_loop)
    
    async def _monitoring_loop(self):
        """Main monitoring loop that sends data to dashboard"""
        while self.is_connected:
            try:
                if self.health_monitor:
                    # Capture current snapshot
                    snapshot = await self.health_monitor._capture_snapshot()
                    
                    # Send to dashboard
                    await self.sio.emit('health_snapshot', snapshot)
                    
                    # Send individual alerts
                    for alert in snapshot.get('active_alerts', []):
                        await self.sio.emit('alert', alert)
                
                await trio.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await trio.sleep(5.0)
    
    async def _simulation_loop(self):
        """Simulate neural activity for demonstration"""
        step = 0
        while self.is_connected and self.network:
            try:
                # Simulate processing some data
                test_features = np.random.randn(384).astype(np.float32)
                test_text = f"Test query {step + 1}: What is the meaning of life?"
                
                # Process through network
                result = await self.network.process_data(test_features, test_text)
                
                # Record firing events if health monitor is available
                if self.health_monitor:
                    # Simulate some firing events
                    for i in range(3):  # Simulate 3 neurons firing
                        neuron_id = f"neuron_{i}_{step}"
                        region = ['thalamus', 'hippocampus', 'amygdala'][i % 3]
                        
                        firing_event = {
                            'timestamp': time.time(),
                            'neuron_id': neuron_id,
                            'region': region,
                            'firing_strength': np.random.random(),
                            'trigger_source': 'simulation',
                            'trigger_details': f'Step {step} processing',
                            'context': test_text
                        }
                        
                        self.health_monitor.record_firing_event(firing_event)
                        await self.sio.emit('firing_event', firing_event)
                
                step += 1
                await trio.sleep(3.0)  # Process every 3 seconds
                
            except Exception as e:
                print(f"❌ Error in simulation loop: {e}")
                await trio.sleep(5.0)
    
    async def run(self):
        """Main run method"""
        print("🚀 Starting AURA WebSocket Bridge...")
        
        # Connect to WebSocket server
        if not await self.connect_to_server():
            return
        
        # Initialize AURA network
        if not await self.initialize_aura_network():
            return
        
        # Start health monitoring
        await self.start_health_monitoring()
    
    async def stop(self):
        """Stop the bridge"""
        print("🛑 Stopping AURA WebSocket Bridge...")
        self.is_connected = False
        await self.sio.disconnect()


async def main():
    """Main entry point"""
    bridge = AURAWebSocketBridge()
    
    try:
        await bridge.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    finally:
        await bridge.stop()


if __name__ == "__main__":
    trio.run(main)
