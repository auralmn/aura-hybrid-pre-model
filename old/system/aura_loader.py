import argparse
import importlib.util
import os
import sys
import trio


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


async def run_svc(mode_args):
    mod = load_module("net_runner_mod", os.path.join(os.getcwd(), "net-runner.py"))
    runner = mod.SVCNetworkRunner(
        dataset_path=mode_args.dataset,
        offline=mode_args.offline,
        samples=mode_args.samples,
        nlms_clamp=None if mode_args.no_clamp else (0.0, 1.0),
        nlms_l2=mode_args.nlms_l2,
        features_mode=mode_args.features_mode,
        features_alpha=mode_args.features_alpha,
        weights_dir=mode_args.weights_dir,
        epochs=mode_args.epochs,
        plot_out=mode_args.plot_out,
        metrics_csv=mode_args.metrics_csv,
        startnew=mode_args.startnew,
    )
    ok = await runner.run_complete_pipeline()
    return ok


async def run_chat(interactive: bool):
    import chat_runner as cr
    if interactive:
        await cr.run_chat_interactive()
    else:
        await cr.run_chat_demo()


async def run_historical():
    import historical_runner as hr
    await hr.demo_historical()


async def run_cns():
    import cns_runner as cnr
    await cnr.run_cns_demo()


def main():
    p = argparse.ArgumentParser(description="Aura Loader — unified system loop")
    sub = p.add_subparsers(dest="mode", required=True)

    # SVC
    sp = sub.add_parser("svc", help="Run SVC training pipeline")
    sp.add_argument("--dataset", type=str, default=os.getenv('AURA_SVC_DATASET'))
    sp.add_argument("--offline", action="store_true", default=os.getenv('AURA_OFFLINE','0').lower() in ('1','true','yes'))
    sp.add_argument("--samples", type=int, default=int(os.getenv('AURA_SVC_SAMPLES','500')))
    sp.add_argument("--epochs", type=int, default=int(os.getenv('AURA_EPOCHS','1')))
    sp.add_argument("--weights-dir", type=str, default=os.getenv('AURA_WEIGHTS_DIR','svc_nlms_weights'))
    sp.add_argument("--startnew", action="store_true", default=os.getenv('AURA_STARTNEW','0').lower() in ('1','true','yes'))
    sp.add_argument("--no-clamp", action="store_true", help="Disable NLMS clamping")
    sp.add_argument("--nlms-l2", type=float, default=float(os.getenv('AURA_NLMS_L2','1e-4')))
    sp.add_argument("--features-mode", choices=['sbert','phasor','combined'], default=os.getenv('AURA_FEATURES_MODE','sbert'))
    sp.add_argument("--features-alpha", type=float, default=float(os.getenv('AURA_FEATURES_ALPHA','0.7')))
    sp.add_argument("--plot-out", type=str, default=os.getenv('AURA_PLOT_OUT','training_metrics.png'))
    sp.add_argument("--metrics-csv", type=str, default=os.getenv('AURA_METRICS_CSV','training_metrics.csv'))

    # Chat
    cp = sub.add_parser("chat", help="Run chat router demo")
    cp.add_argument("--interactive", action="store_true")
    cp.add_argument("--qdrant", action="store_true", help="Stream routing to Qdrant (env AURA_QDRANT_STREAM=1)")
    cp.add_argument("--url", type=str, default=os.getenv('QDRANT_URL','http://localhost'))
    cp.add_argument("--port", type=int, default=int(os.getenv('QDRANT_PORT','6333')))

    # Historical
    hp = sub.add_parser("historical", help="Run historical demo")

    # CNS
    np = sub.add_parser("cns", help="Run CNS orchestration demo")

    # Qdrant snapshot
    qp = sub.add_parser("qdrant", help="Snapshot network map to Qdrant")
    qp.add_argument("--url", type=str, default=os.getenv('QDRANT_URL','http://localhost'))
    qp.add_argument("--port", type=int, default=int(os.getenv('QDRANT_PORT','6333')))

    # Augment historical dataset with canonical periods
    ah = sub.add_parser("augment-hist", help="Augment historical JSONL with period names")
    ah.add_argument("--input", required=True)
    ah.add_argument("--output", required=True)
    ah.add_argument("--teacher", type=str, default="historical_teacher.md")

    # Conversation training
    tp = sub.add_parser("conv-train", help="Process conversation JSONL dataset for routing/topic relevance")
    tp.add_argument("--data", type=str, required=True, help="Path to conversation JSONL")
    tp.add_argument("--limit", type=int, default=0)

    # Train-all orchestrator
    ap = sub.add_parser("train-all", help="Scan datasets/* and run appropriate trainers")
    ap.add_argument("--root", type=str, default="datasets", help="Root folder to scan for *.jsonl")
    ap.add_argument("--offline", action="store_true", default=os.getenv('AURA_OFFLINE','0').lower() in ('1','true','yes'))
    ap.add_argument("--samples", type=int, default=int(os.getenv('AURA_SVC_SAMPLES','500')))
    ap.add_argument("--epochs", type=int, default=int(os.getenv('AURA_EPOCHS','1')))
    ap.add_argument("--limit", type=int, default=0, help="Max items for conv-train per file (0=all)")
    ap.add_argument("--teacher", type=str, default="historical_teacher.md", help="Path to historical teacher markdown")

    args = p.parse_args()

    if args.mode == 'svc':
        trio.run(run_svc, args)
    elif args.mode == 'chat':
        if getattr(args, 'qdrant', False):
            os.environ['AURA_QDRANT_STREAM'] = '1'
            os.environ['QDRANT_URL'] = str(args.url)
            os.environ['QDRANT_PORT'] = str(args.port)
        trio.run(run_chat, args.interactive)
    elif args.mode == 'historical':
        trio.run(run_historical)
    elif args.mode == 'cns':
        trio.run(run_cns)
    elif args.mode == 'qdrant':
        # Minimal snapshot without trio
        from network import Network
        from qdrant_mapper import QdrantMapper
        net = Network()
        # No need to init NLMS for a static snapshot
        mapper = QdrantMapper(url=args.url, port=args.port)
        mapper.ensure_collections()
        mapper.snapshot_network(net)
    elif args.mode == 'augment-hist':
        from tools.augment_historical_periods import run as augment_run
        res = augment_run(args.input, args.output, args.teacher)
        print(res)
    elif args.mode == 'conv-train':
        import trio as _trio
        async def _run():
            from network import Network
            from conv_trainer import AuraConversationTrainer
            net = Network()
            await net.init_weights()
            trainer = AuraConversationTrainer(net)
            res = await trainer.process_dataset(args.data, limit=args.limit or None)
            print(res)
        _trio.run(_run)
    elif args.mode == 'train-all':
        import glob
        import asyncio as _asyncio
        # Collect JSONL files under root
        files = glob.glob(os.path.join(args.root, '**', '*.jsonl'), recursive=True)
        if not files:
            print(f"No JSONL files found under {args.root}")
            return
        print(f"Found {len(files)} dataset files under {args.root}")
        # Decide per file which trainer to run
        for fp in files:
            name = os.path.basename(fp).lower()
            path_lower = fp.lower()
            print(f"\n➡️  Processing {fp}")
            if 'svc' in name or 'svc' in path_lower:
                # Use SVC pipeline
                class Obj: pass
                o = Obj()
                o.dataset = fp
                o.offline = args.offline
                o.samples = args.samples
                o.epochs = args.epochs
                o.weights_dir = os.getenv('AURA_WEIGHTS_DIR','svc_nlms_weights')
                o.startnew = False
                o.no_clamp = False
                o.nlms_l2 = float(os.getenv('AURA_NLMS_L2','1e-4'))
                o.features_mode = os.getenv('AURA_FEATURES_MODE','sbert')
                o.features_alpha = float(os.getenv('AURA_FEATURES_ALPHA','0.7'))
                o.plot_out = None
                o.metrics_csv = None
                try:
                    import trio
                    trio.run(run_svc, o)
                except Exception as e:
                    print(f"SVC trainer failed on {fp}: {e}")
            elif 'historical' in name or 'historical' in path_lower:
                # Historical trainer
                import trio as _trio
                async def _run_hist(path: str):
                    from historical_trainer import HistoricalTrainer
                    # Optionally load teacher (currently unused, placeholder)
                    tr = HistoricalTrainer(offline=args.offline, teacher_path=args.teacher)
                    res = await tr.train_file(path, limit=(args.limit or None))
                    print(res)
                try:
                    _trio.run(_run_hist, fp)
                except Exception as e:
                    print(f"Historical trainer failed on {fp}: {e}")
            elif 'empathy' in name or 'emotion' in name or 'empathy' in path_lower or 'emotion' in path_lower:
                # Conversation trainer focusing on emotions
                import trio as _trio
                async def _run_one(path: str):
                    from network import Network
                    from conv_trainer import AuraConversationTrainer
                    net = Network()
                    await net.init_weights()
                    trainer = AuraConversationTrainer(net)
                    # Route missing/unknown topics to 'emotions' for empathy files
                    res = await trainer.process_dataset(path, limit=(args.limit or None), default_topic='emotions')
                    print(res)
                try:
                    _trio.run(_run_one, fp)
                except Exception as e:
                    print(f"Conv-train (empathy) failed on {fp}: {e}")
            else:
                # Treat as conversation dataset
                import trio as _trio
                async def _run_one(path: str):
                    from network import Network
                    from conv_trainer import AuraConversationTrainer
                    net = Network()
                    await net.init_weights()
                    trainer = AuraConversationTrainer(net)
                    res = await trainer.process_dataset(path, limit=(args.limit or None))
                    print(res)
                try:
                    _trio.run(_run_one, fp)
                except Exception as e:
                    print(f"Conv-train failed on {fp}: {e}")


if __name__ == "__main__":
    main()
