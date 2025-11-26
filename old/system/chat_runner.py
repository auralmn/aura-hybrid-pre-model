import trio
import numpy as np
import argparse
import os

from ..core.network import Network
from ..utils.chat_orchestrator import ChatOrchestrator


async def run_chat_demo():
    print("🗣️  Chat Routing Demo")
    net = Network()
    await net.init_weights()

    # Sample queries to exercise the router
    queries = [
        "Hi, how are you?",
        "Tell me about the Roman Empire",
        "I'm worried about climate change",
        "What did we talk about yesterday?",
        "Analyze the causes and effects of World War II",
    ]

    # Use zero features (384-d) to avoid requiring SBERT for this demo
    feat = np.zeros(384, dtype=np.float32)
    # Optional Qdrant streaming
    streamer = None
    if os.getenv('AURA_QDRANT_STREAM', '0').lower() in ('1','true','yes'):
        try:
            from qdrant_stream import QdrantStreamer  # type: ignore
            streamer = QdrantStreamer()
        except Exception:
            streamer = None

    for q in queries:
        decision = net._thalamic_router.analyze_conversation_intent(q, feat)
        plan = net._thalamic_router.route_conversation(decision, q, feat)
        expl = net._thalamic_router.explain_routing_decision(decision)
        print("\nQ:", q)
        print("Plan:", plan)
        print("Why:", expl)
        # Qdrant stream routing decision
        if streamer is not None:
            try:
                streamer.upsert_routing(decision, feat)
            except Exception:
                pass
        # Pretend the response was decent to reinforce routing a bit
        outcome = {"user_satisfaction": 0.8, "response_quality": 0.7}
        await net._thalamic_router.adaptive_routing_update(plan, outcome, feat)

    print("\nRouting stats:")
    print(net._thalamic_router.get_routing_statistics())


async def run_chat_interactive():
    print("🗣️  Chat Router Interactive (type 'exit' to quit)")
    net = Network()
    await net.init_weights()
    feat = np.zeros(384, dtype=np.float32)
    streamer = None
    if os.getenv('AURA_QDRANT_STREAM', '0').lower() in ('1','true','yes'):
        try:
            from qdrant_stream import QdrantStreamer  # type: ignore
            streamer = QdrantStreamer()
        except Exception:
            streamer = None

    async def ainput(prompt: str = "") -> str:
        return await trio.to_thread.run_sync(lambda: input(prompt))

    orch = ChatOrchestrator(net=net, offline=True)
    while True:
        q = await ainput("You> ")
        if not q or q.strip().lower() in {"exit", "quit"}:
            break
        res = await orch.respond(q)
        print("Aura>", res['response_text'])
        print("Why:", res['routing_explanation'])
        # Qdrant stream routing
        if streamer is not None:
            try:
                streamer.upsert_routing(res['routing_plan'], feat)
            except Exception:
                pass
    print("\nRouting stats:")
    print(net._thalamic_router.get_routing_statistics())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat router demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat routing loop")
    args = parser.parse_args()
    if args.interactive:
        trio.run(run_chat_interactive)
    else:
        trio.run(run_chat_demo)
