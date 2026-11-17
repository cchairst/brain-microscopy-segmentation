"""Command-line interface. Run `brainseg --help`."""

import argparse
import json

from brainseg.methods.classical import METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(prog='brainseg', description='Brain microscopy tissue segmentation.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', help='Train a k-means or random forest baseline on manifest train rows.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--method', choices=['kmeans', 'rf'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pixels', type=int, default=20000)

    p = sub.add_parser('predict', help='Predict and score a manifest split with a trained model.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--output', required=True)

    p = sub.add_parser(
        'segment', help='Unsupervised classical segmentation of a manifest split, scored where masks exist.'
    )
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=METHODS, default='gmm', help='Gray/white split')
    p.add_argument('--retardance', type=int, default=0, help='Index of the retardance image in each row')
    p.add_argument('--reflectivity', type=int, default=-1, help='Index of the reflectivity image in each row')
    p.add_argument(
        '--features',
        type=lambda s: [int(i) for i in s.split(',')],
        help='Comma-separated image indices to cluster on, gmm/kmeans only (default: retardance only)',
    )
    p.add_argument('--pixel-size', type=float, default=0.055, help='Pixel size in mm (0.055 for 10x low-res sections)')
    p.add_argument('--min-area', type=float, default=0.2, help='Smallest tissue or gray/white island kept, in mm^2')
    p.add_argument('--min-fluid-area', type=float, default=1.2, help='Smallest enclosed hole labeled fluid, in mm^2')
    p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('demo', help='Synthetic end-to-end software check.')
    p.add_argument('--output', default='outputs/synthetic_demo')

    p = atropos = sub.add_parser('atropos', help='ANTs Atropos on a 2D section or 3D NIfTI (intensity classes only).')
    p.add_argument('--image', required=True, help='2D grayscale PNG/JPEG or 2D/3D NIfTI')
    p.add_argument('--mask', help='Reviewed binary mask on the processed grid; NIfTI or lossless image')
    p.add_argument('--output', required=True)
    p.add_argument('--max-size', type=int, default=1100, help='Max JPEG/PNG side for a downsampled pilot')
    p.add_argument('--classes', type=int, default=3)
    p.add_argument(
        '--threshold', type=float, default=20, help='Unreviewed foreground mask threshold on 0-255 image inputs'
    )
    p.add_argument('--mrf', type=float, default=0.2)

    p = sub.add_parser('vlad-pilot', help='Single- vs three-contrast 3D Atropos on MRI-aligned Vlad volumes.')
    p.add_argument('--data', default='data/vlad')
    p.add_argument('--output', default='outputs/vlad_3d_pilot')
    p.add_argument('--factor', type=int, default=2)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--mask', help='Optional binary native-grid MRI brain mask (not tissue ground truth)')

    args = parser.parse_args(argv)
    # Imports are deferred so the base install works without the optional ANTs dependencies.
    if args.command == 'train':
        from brainseg.methods.baselines import train

        train(args.manifest, args.method, args.output, args.seed, args.pixels)
        print(f'Model saved to {args.output}')
    elif args.command == 'predict':
        from brainseg.benchmark import run_predictions

        report = run_predictions(args.manifest, args.model, args.split, args.output)
        print(f'Report saved to {args.output}/report.json; mean sample Dice: {report["mean_sample_macro_dice"]}')
    elif args.command == 'segment':
        from brainseg.benchmark import run_classical

        report = run_classical(
            args.manifest,
            args.split,
            args.output,
            args.method,
            retardance=args.retardance,
            reflectivity=args.reflectivity,
            features=args.features,
            pixel_size=args.pixel_size,
            min_area=args.min_area,
            min_fluid_area=args.min_fluid_area,
            seed=args.seed,
        )
        print(f'Report saved to {args.output}/report.json; mean sample Dice: {report["mean_sample_macro_dice"]}')
    elif args.command == 'demo':
        from brainseg.benchmark import demo

        demo(args.output)
    elif args.command == 'atropos':
        if args.classes < 2 or args.max_size < 32 or args.mrf < 0:
            atropos.error('classes >= 2, max-size >= 32, and mrf >= 0 are required')
        from brainseg.methods.atropos import segment

        report = segment(args.image, args.output, args.mask, args.max_size, args.classes, args.threshold, args.mrf)
        print(json.dumps(report, indent=2))
    else:
        from brainseg.vlad_pilot import run

        run(args.data, args.output, args.factor, args.iterations, args.mask)


if __name__ == '__main__':
    main()
