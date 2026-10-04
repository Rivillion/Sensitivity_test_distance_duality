"""Sample the three Section IV.B fits with Gaussian external calibrations."""
from src.release_io import verify_inputs
from src.sample_section4b import main

if __name__ == "__main__":
    verify_inputs()
    raise SystemExit(main())
