import trio
from ..core.network import Network


async def run_cns_demo():
    print("🧠 CNS Orchestration Demo")
    net = Network()
    await net.init_weights()

    # Initial assessment
    state = net._cns.assess_global_state()
    print("Initial CNS state:", state)

    # Coordinate for a chat context
    res1 = await net._cns.coordinate({'type': 'chat', 'urgency': 0.3})
    print("Chat context:", res1)

    # Coordinate for SVC task under mild threat
    res2 = await net._cns.coordinate({'type': 'svc', 'urgency': 0.5, 'threat': 0.6})
    print("SVC context (threat=0.6):", res2)

    # Coordinate for emotional scenario with high threat
    res3 = await net._cns.coordinate({'type': 'emotion', 'urgency': 0.8, 'threat': 0.9, 'multi_topic': True})
    print("Emotion context (threat=0.9):", res3)

    # Final assessment
    state2 = net._cns.assess_global_state()
    print("Final CNS state:", state2)


if __name__ == '__main__':
    trio.run(run_cns_demo)

