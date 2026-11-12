"""Usage: python -m rhm_probing --reference ... --checkpoint ... --output ..."""
import argparse
from .runner import DEFAULT_PROBES, probe_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Linear probes of VT-RHM checkpoint latents")
    parser.add_argument("--reference", required=True, help="Original probe metadata bundle (.pkl)")
    parser.add_argument("--checkpoint", required=True, action="append", help="Model checkpoint (.pt); repeat for multiple steps")
    parser.add_argument("--output", required=True, help="Destination probe bundle (.pkl)")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--probe", action="append", choices=DEFAULT_PROBES, help="Select probes (default: all three)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    probe_checkpoints(args.reference, args.checkpoint, args.output,
                      num_heads=args.num_heads, probes=args.probe or DEFAULT_PROBES,
                      batch_size=args.batch_size, device=args.device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
