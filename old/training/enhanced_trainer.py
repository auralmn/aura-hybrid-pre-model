"""
Enhanced AURA Trainer - Comprehensive training system with directory support
Combines all training functionality into a single, powerful module
"""

import argparse
import json
import os
import trio
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable

from .directory_loader import DirectoryLoader, discover_and_preview


class EnhancedAuraTrainer:
    """Enhanced trainer that can process multiple files in a directory and single files"""
    
    def __init__(self, trainer_class: Type):
        self.trainer_class = trainer_class
        self.loader = None
    
    async def train_from_directory(
        self,
        directory: str,
        file_pattern: str = "*",
        categories: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        **trainer_kwargs
    ) -> Dict[str, Any]:
        """Train on all files in a directory"""
        self.loader = DirectoryLoader(directory)
        
        # Get categorized files
        all_categories = self.loader.categorize_files()
        
        # Filter by requested categories
        if categories:
            files_to_process = []
            for category in categories:
                if category in all_categories:
                    files_to_process.extend(all_categories[category])
        else:
            files_to_process = []
            for files in all_categories.values():
                files_to_process.extend(files)
        
        # Apply pattern filter
        if file_pattern != "*":
            pattern_path = Path(file_pattern)
            files_to_process = [
                f for f in files_to_process 
                if pattern_path.match(f.name)
            ]
        
        # Limit number of files
        if max_files:
            files_to_process = files_to_process[:max_files]
        
        print(f"🎯 Processing {len(files_to_process)} files...")
        
        # Process files
        results = []
        for file_path in files_to_process:
            try:
                print(f"🔄 Processing {file_path.name}...")
                
                # Create trainer instance
                trainer = self.trainer_class(**trainer_kwargs)
                
                # Determine training method based on file category
                category = self._categorize_single_file(file_path)
                
                if hasattr(trainer, 'process_movie_scenes_dataset') and category == 'emotional':
                    result = await trainer.process_movie_scenes_dataset(str(file_path))
                elif hasattr(trainer, 'process_dataset'):
                    result = await trainer.process_dataset(str(file_path))
                else:
                    result = {'error': 'Unknown trainer method', 'events_processed': 0}
                
                result['source_file'] = str(file_path)
                result['file_category'] = category
                results.append(result)
                
                print(f"✅ Completed {file_path.name}: {result.get('events_processed', result.get('total_scenes_processed', 0))} events")
                
            except Exception as e:
                error_result = {
                    'source_file': str(file_path),
                    'error': str(e),
                    'events_processed': 0
                }
                results.append(error_result)
                print(f"❌ Failed {file_path.name}: {e}")
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _categorize_single_file(self, file_path: Path) -> str:
        """Categorize a single file"""
        filename = file_path.name.lower()
        
        if any(keyword in filename for keyword in ['hist', 'historical', 'history']):
            return 'historical'
        elif any(keyword in filename for keyword in ['emotion', 'movie', 'scene']):
            return 'emotional'
        elif any(keyword in filename for keyword in ['conv', 'chat', 'dialogue']):
            return 'conversational'
        elif any(keyword in filename for keyword in ['causal', 'cause', 'effect']):
            return 'causal'
        elif any(keyword in filename for keyword in ['socratic', 'question', 'qa']):
            return 'socratic'
        else:
            return 'general'
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple files"""
        total_events = sum(r.get('events_processed', r.get('total_scenes_processed', 0)) for r in results)
        total_files = len(results)
        successful_files = len([r for r in results if 'error' not in r])
        failed_files = total_files - successful_files
        
        # Category breakdown
        category_stats = {}
        for result in results:
            category = result.get('file_category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'files': 0, 'events': 0}
            category_stats[category]['files'] += 1
            category_stats[category]['events'] += result.get('events_processed', result.get('total_scenes_processed', 0))
        
        return {
            'summary': {
                'total_files_processed': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'total_events_processed': total_events,
                'success_rate': successful_files / total_files if total_files > 0 else 0
            },
            'category_breakdown': category_stats,
            'file_results': results
        }


def print_directory_results(results: Dict[str, Any]):
    """Print formatted directory training results"""
    print("\n🎉 DIRECTORY TRAINING COMPLETE")
    print("=" * 50)
    summary = results['summary']
    print(f"📊 Total Files Processed: {summary['total_files_processed']}")
    print(f"✅ Successful: {summary['successful_files']}")
    print(f"❌ Failed: {summary['failed_files']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1%}")
    print(f"🎯 Total Events: {summary['total_events_processed']}")
    
    print("\n📋 By Category:")
    for category, stats in results['category_breakdown'].items():
        print(f"   {category}: {stats['files']} files, {stats['events']} events")


# Command Handlers
def cmd_discover(args):
    """Discover and preview files in directory"""
    report = discover_and_preview(args.directory)
    
    if args.save_report:
        report_file = f"discovery_report_{Path(args.directory).name}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Discovery report saved to: {report_file}")


def cmd_hist_dir(args):
    """Run historical trainer on all files in directory"""
    from historical_trainer import AuraHistoricalTrainer
    from ..core.network import Network
    
    async def run_directory_training():
        if not args.no_preview:
            discover_and_preview(args.directory)
        
        # Setup network and enhanced trainer
        class HistTrainerWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            async def process_dataset(self, file_path):
                net = Network()
                if not self.kwargs.get('no_init'):
                    await net.init_weights()
                
                trainer = AuraHistoricalTrainer(
                    net,
                    offline=self.kwargs.get('offline', False),
                    device=self.kwargs.get('device'),
                    verbose=self.kwargs.get('verbose', False),
                    weights_dir=self.kwargs.get('weights_dir', 'svc_nlms_weights'),
                    epochs=self.kwargs.get('epochs', 1),
                    mu_tok=self.kwargs.get('mu_tok', 0.08),
                    mu_bias=self.kwargs.get('mu_bias', 0.02),
                    l2=self.kwargs.get('l2', 1e-3),
                    no_hints=self.kwargs.get('no_hints', False),
                    hint_threshold=self.kwargs.get('hint_threshold', 0.8),
                    balance=self.kwargs.get('balance', False),
                    freeze_router=self.kwargs.get('freeze_router', False)
                )
                return await trainer.process_dataset(file_path)
        
        enhanced_trainer = EnhancedAuraTrainer(HistTrainerWrapper)
        
        results = await enhanced_trainer.train_from_directory(
            directory=args.directory,
            file_pattern=args.pattern,
            categories=args.categories or ['historical', 'general'],
            max_files=args.max_files,
            offline=args.offline,
            device=args.device,
            verbose=args.verbose,
            weights_dir=args.weights_dir,
            epochs=args.epochs,
            mu_tok=args.mu_tok,
            mu_bias=args.mu_bias,
            l2=args.l2,
            no_hints=args.no_hints,
            hint_threshold=args.hint_threshold,
            balance=args.balance,
            freeze_router=args.freeze_router,
            no_init=args.no_init
        )
        
        print_directory_results(results)
        
        # Save results
        if args.save_results:
            results_file = f"historical_results_{Path(args.directory).name}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {results_file}")
    
    trio.run(run_directory_training)


def cmd_movie_emotional_dir(args):
    """Run movie emotional trainer on all files in directory"""
    from movie_emotional_trainer import AuraMovieEmotionalTrainer
    from ..core.network import Network
    
    async def run_directory_training():
        if not args.no_preview:
            discover_and_preview(args.directory)
        
        # Create network wrapper that handles attention setup
        class MovieTrainerWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            async def process_movie_scenes_dataset(self, file_path):
                net = Network()
                if self.kwargs.get('enable_attention'):
                    net.enable_attention_learning(['thalamus', 'hippocampus'])
                if not self.kwargs.get('no_init'):
                    await net.init_weights()
                
                trainer = AuraMovieEmotionalTrainer(
                    net, 
                    enable_attention=self.kwargs.get('enable_attention', False)
                )
                return await trainer.process_movie_scenes_dataset(file_path)
        
        enhanced_trainer = EnhancedAuraTrainer(MovieTrainerWrapper)
        
        results = await enhanced_trainer.train_from_directory(
            directory=args.directory,
            file_pattern=args.pattern,
            categories=['emotional', 'general'],  # Focus on emotional files
            max_files=args.max_files,
            enable_attention=args.enable_attention,
            no_init=args.no_init
        )
        
        print_directory_results(results)
        
        # Save results
        if args.save_results:
            results_file = f"movie_emotional_results_{Path(args.directory).name}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {results_file}")
    
    trio.run(run_directory_training)


def cmd_conv_dir(args):
    """Run conversation trainer on all files in directory"""
    from conv_trainer import AuraConversationTrainer
    from ..core.network import Network
    
    async def run_directory_training():
        if not args.no_preview:
            discover_and_preview(args.directory)
        
        # Setup network and enhanced trainer
        class ConvTrainerWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            async def process_dataset(self, file_path):
                net = Network()
                if not self.kwargs.get('no_init'):
                    await net.init_weights()
                
                trainer = AuraConversationTrainer(net)
                return await trainer.process_dataset(
                    file_path, 
                    limit=self.kwargs.get('limit')
                )
        
        enhanced_trainer = EnhancedAuraTrainer(ConvTrainerWrapper)
        
        results = await enhanced_trainer.train_from_directory(
            directory=args.directory,
            file_pattern=args.pattern,
            categories=['conversational', 'general'],
            max_files=args.max_files,
            no_init=args.no_init,
            limit=args.limit
        )
        
        print_directory_results(results)
    
    trio.run(run_directory_training)


def cmd_hist_edu(args):
    """Run historical education trainer on a conversation JSONL"""
    from historical_education_trainer import AuraHistoricalEducationTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraHistoricalEducationTrainer(net, weights_dir=args.weights_dir)
        result = await trainer.process_dataset(args.data)
        print(f"✅ Historical education training complete: {result}")
    
    trio.run(run_training)


def cmd_causal(args):
    """Run causal history trainer on a causal events JSONL"""
    from causal_trainer import AuraCausalTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraCausalTrainer(net)
        result = await trainer.process_dataset(args.data)
        print(f"✅ Causal training complete: {result}")
    
    trio.run(run_training)


def cmd_conv(args):
    """Run conversation trainer for routing/topic relevance"""
    from conv_trainer import AuraConversationTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraConversationTrainer(net)
        result = await trainer.process_dataset(args.data, limit=args.limit)
        print(f"✅ Conversation training complete: {result}")
    
    trio.run(run_training)


def cmd_socratic(args):
    """Train numeric regressor on Socratic QA JSONL (question/answer)"""
    from socratic_trainer import AuraSocraticTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraSocraticTrainer(
            net,
            offline=args.offline,
            device=args.device,
            epochs=args.epochs,
            quiet=args.quiet,
            weights_dir=args.weights_dir
        )
        result = await trainer.process_dataset(args.data, limit=args.limit)
        print(f"✅ Socratic training complete: {result}")
    
    trio.run(run_training)


def cmd_hist(args):
    """Run weakly supervised historical era trainer"""
    from historical_trainer import AuraHistoricalTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraHistoricalTrainer(
            net,
            teacher_file=args.teacher,
            offline=args.offline,
            device=args.device,
            verbose=args.verbose,
            weights_dir=args.weights_dir,
            epochs=args.epochs,
            mu_tok=args.mu_tok,
            mu_bias=args.mu_bias,
            l2=args.l2,
            no_hints=args.no_hints,
            hint_threshold=args.hint_threshold,
            balance=args.balance,
            freeze_router=args.freeze_router
        )
        result = await trainer.process_dataset(args.data, limit=args.limit)
        print(f"✅ Historical training complete: {result}")
    
    trio.run(run_training)


def cmd_movie_emotional(args):
    """Train emotional intelligence on movie scene annotations"""
    from movie_emotional_trainer import AuraMovieEmotionalTrainer
    from ..core.network import Network
    
    async def run_training():
        net = Network()
        if args.enable_attention:
            net.enable_attention_learning(['thalamus', 'hippocampus'])
        if not args.no_init:
            await net.init_weights()
        
        trainer = AuraMovieEmotionalTrainer(net, enable_attention=args.enable_attention)
        result = await trainer.process_movie_scenes_dataset(args.data)
        print(f"✅ Movie emotional training complete: {result}")
    
    trio.run(run_training)


def add_sbert_movie_commands(sub):
    """Add SBERT movie commands to argument parser"""
    # This function can be implemented if needed for specific SBERT functionality
    pass


def main():
    """Main entry point for the enhanced trainer"""
    p = argparse.ArgumentParser(description="Enhanced AURA Trainer - Comprehensive training system")
    sub = p.add_subparsers(dest="cmd", required=True)

    # === DIRECTORY COMMANDS ===
    
    # Directory discovery command
    pd = sub.add_parser("discover", help="Discover and preview files in directory")
    pd.add_argument("--directory", "--data", dest="directory", required=True, help="Directory to scan")
    pd.add_argument("--save-report", action="store_true", help="Save discovery report to JSON")
    pd.set_defaults(func=cmd_discover)

    add_sbert_movie_commands(sub)

    # Directory-based historical trainer
    phd = sub.add_parser("hist-dir", help="Run historical trainer on all files in directory")
    phd.add_argument("--directory", required=True, help="Directory containing training files")
    phd.add_argument("--pattern", default="*", help="File name pattern filter")
    phd.add_argument("--categories", nargs="*", choices=['historical', 'emotional', 'conversational', 'causal', 'socratic', 'general'], 
                     help="File categories to process")
    phd.add_argument("--max-files", type=int, help="Maximum number of files to process")
    phd.add_argument("--no-preview", action="store_true", help="Skip file preview")
    phd.add_argument("--auto-confirm", action="store_true", help="Auto-confirm training start")
    phd.add_argument("--save-results", action="store_true", help="Save detailed results to JSON")
    
    # Historical trainer parameters
    phd.add_argument("--offline", action="store_true", help="Run without SBERT embeddings")
    phd.add_argument("--device", type=str, default=None, help="Device for SBERT")
    phd.add_argument("--verbose", action="store_true", help="Show progress")
    phd.add_argument("--weights-dir", type=str, default="svc_nlms_weights")
    phd.add_argument("--epochs", type=int, default=1)
    phd.add_argument("--mu-tok", type=float, default=0.08, dest="mu_tok")
    phd.add_argument("--mu-bias", type=float, default=0.02, dest="mu_bias")
    phd.add_argument("--l2", type=float, default=1e-3)
    phd.add_argument("--no-hints", action="store_true")
    phd.add_argument("--hint-threshold", type=float, default=0.8)
    phd.add_argument("--balance", action="store_true")
    phd.add_argument("--freeze-router", action="store_true")
    phd.set_defaults(func=cmd_hist_dir)

    # Directory-based movie emotional trainer
    pmd = sub.add_parser("movie-emotional-dir", help="Run movie emotional trainer on all files in directory")
    pmd.add_argument("--directory", required=True, help="Directory containing movie/emotional files")
    pmd.add_argument("--pattern", default="*", help="File name pattern filter")
    pmd.add_argument("--max-files", type=int, help="Maximum number of files to process")
    pmd.add_argument("--no-preview", action="store_true", help="Skip file preview")
    pmd.add_argument("--auto-confirm", action="store_true", help="Auto-confirm training start")
    pmd.add_argument("--save-results", action="store_true", help="Save detailed results to JSON")
    pmd.add_argument("--enable-attention", action="store_true", help="Enable spiking attention")
    pmd.add_argument("--no-init", action="store_true", help="Skip weight initialization")
    pmd.set_defaults(func=cmd_movie_emotional_dir)

    # Directory-based conversation trainer
    pcd = sub.add_parser("conv-dir", help="Run conversation trainer on all files in directory")
    pcd.add_argument("--directory", required=True, help="Directory containing conversation files")
    pcd.add_argument("--pattern", default="*", help="File name pattern filter")
    pcd.add_argument("--max-files", type=int, help="Maximum number of files to process")
    pcd.add_argument("--no-preview", action="store_true", help="Skip file preview")
    pcd.add_argument("--limit", type=int, default=0)
    pcd.add_argument("--no-init", action="store_true")
    pcd.set_defaults(func=cmd_conv_dir)

    # === SINGLE FILE COMMANDS ===

    pe = sub.add_parser("hist-edu", help="Run historical education trainer on a conversation JSONL")
    pe.add_argument("--data", required=True, help="Path to historical conversations JSONL")
    pe.add_argument("--no-init", action="store_true", help="Skip network weight initialization")
    pe.add_argument("--weights-dir", type=str, default="svc_nlms_weights")
    pe.set_defaults(func=cmd_hist_edu)

    pc = sub.add_parser("causal", help="Run causal history trainer on a causal events JSONL")
    pc.add_argument("--data", required=True, help="Path to causal events JSONL")
    pc.add_argument("--no-init", action="store_true", help="Skip network weight initialization")
    pc.set_defaults(func=cmd_causal)

    pv = sub.add_parser("conv", help="Run conversation trainer for routing/topic relevance")
    pv.add_argument("--data", required=True)
    pv.add_argument("--limit", type=int, default=0)
    pv.add_argument("--no-init", action="store_true")
    pv.set_defaults(func=cmd_conv)

    ps = sub.add_parser("socratic", help="Train numeric regressor on Socratic QA JSONL (question/answer)")
    ps.add_argument("--data", required=True)
    ps.add_argument("--limit", type=int, default=0)
    ps.add_argument("--offline", action="store_true")
    ps.add_argument("--device", type=str, default=None)
    ps.add_argument("--epochs", type=int, default=1)
    ps.add_argument("--quiet", action="store_true")
    ps.add_argument("--weights-dir", type=str, default="svc_nlms_weights")
    ps.set_defaults(func=cmd_socratic)

    ph = sub.add_parser("hist", help="Run weakly supervised historical era trainer")
    ph.add_argument("--data", required=True)
    ph.add_argument("--teacher", type=str, default="historical_teacher.md")
    ph.add_argument("--limit", type=int, default=0)
    ph.add_argument("--offline", action="store_true", help="Run without SBERT embeddings")
    ph.add_argument("--device", type=str, default=None, help="Device for SBERT (e.g., mps, cuda, cpu)")
    ph.add_argument("--verbose", action="store_true", help="Show progress with running accuracy")
    ph.add_argument("--weights-dir", type=str, default="svc_nlms_weights")
    ph.add_argument("--epochs", type=int, default=1)
    ph.add_argument("--mu-tok", type=float, default=0.08, dest="mu_tok")
    ph.add_argument("--mu-bias", type=float, default=0.02, dest="mu_bias")
    ph.add_argument("--l2", type=float, default=1e-3)
    ph.add_argument("--no-hints", action="store_true")
    ph.add_argument("--hint-threshold", type=float, default=0.8)
    ph.add_argument("--balance", action="store_true")
    ph.add_argument("--freeze-router", action="store_true")
    ph.set_defaults(func=cmd_hist)

    pm = sub.add_parser("movie-emotional", help="Train emotional intelligence on movie scene annotations")
    pm.add_argument("--data", required=True, help="Path to movie scenes JSON")
    pm.add_argument("--enable-attention", action="store_true", help="Enable spiking attention")
    pm.add_argument("--no-init", action="store_true", help="Skip weight initialization")
    pm.set_defaults(func=cmd_movie_emotional)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
