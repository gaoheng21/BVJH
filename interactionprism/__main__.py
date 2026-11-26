"""Unified entry point. Subcommand options are delegated to their native parsers."""
import argparse
import importlib
import sys

COMMANDS = {
    'detect': ('coalitions', 'Frozen variable-coalition candidate discovery'),
    'calibrate': ('calibration', 'Frozen E12R3 strict-upper-quantile stage-2 detector'),
    'discover': ('discovery', 'Interaction-guided behavioral rule discovery'),
    'verify': ('verification', 'Layered generation verification'),
    'verify-position': ('position_verification', 'Position-faithful verification'),
    'sweep': ('signature_sweep', 'Minimal signature sweep'),
    'incremental': ('incremental', 'Incremental signature-cache pipeline'),
    'clean-center': ('clean_center', 'Clean-only reference-center analysis (frozen score)'),
    'audit': ('audit', 'Per-attack ASR protocol audit'),
    'purify': ('purification', 'P2 interaction-nulling purification'),
    'quarantine': ('quarantine', 'Build quarantine pools from confirmed signatures'),
    'evaluate': ('evaluation', 'Paper-protocol ASR evaluation'),
    'evaluate-batch': ('batch_evaluation', 'Batch paper-protocol ASR evaluation'),
    'structural': ('structural', 'Structural interaction verification'),
    'structural-control': ('structural_control', 'Structural control arm'),
    'structural-aggregate': ('structural_aggregate', 'Aggregate structural evidence'),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='python -m interactionprism', description='InteractionPrism research pipeline; frozen E12R3 remains the formal detector.')
    sub = parser.add_subparsers(dest='command', required=True)
    for command, (_, description) in COMMANDS.items():
        sub.add_parser(command, help=description, description=description, add_help=False)
    if not args or args[0] in ('-h', '--help'):
        parser.parse_args(args)
        return
    if args[0] not in COMMANDS:
        parser.error(f'unknown command: {args[0]}')
    name, rest = args[0], args[1:]
    module = importlib.import_module('.' + COMMANDS[name][0], __package__)
    # Legacy report-only tools had no parser. Handle help before doing any work.
    if name in ('sweep', 'quarantine', 'structural-control', 'structural-aggregate'):
        p = argparse.ArgumentParser(prog=f'python -m interactionprism {name}', description=COMMANDS[name][1])
        p.parse_args(rest)
    old = sys.argv
    try:
        sys.argv = [f'python -m interactionprism {name}', *rest]
        return module.main()
    finally:
        sys.argv = old


if __name__ == '__main__':
    main()
