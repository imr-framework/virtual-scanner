# flocra-pulseq
Pulseq interpreter for vnegnev's flow-based OCRA (FLOCRA)

# Usage:
TO INSTALL: In the root folder of this repository (Default `flocra-pulseq`), run `pip install -e .`. This will install the package with an editable link. If you need to move the file structure, run `pip uninstall flocra_pulseq`, and reinstall once your files are where you want them.

To import the package: `import flocra_pulseq.interpreter`

Create interpreter with `flocra_pulseq.interpreter.PSInterpreter`. Run `PSInterpreter.interpret()` to get output array and dictionary. Specify a log path to log errors, warnings, and progress.

# Arguments
rf_center (float): RF center (local oscillator frequency) in Hz.

rf_amp_max (float): Default 5e+3 -- System RF amplitude max in Hz.

grad_max (float): Default 1e+6 -- System gradient max in Hz/m.

gx_max (float): Default None -- System X-gradient max in Hz/m. If None, defaults to grad_max.

gy_max (float): Default None -- System Y-gradient max in Hz/m. If None, defaults to grad_max.

gz_max (float): Default None -- System Z-gradient max in Hz/m. If None, defaults to grad_max.

clk_t (float): Default 1/122.88 -- System clock period in us.

tx_t (float): Default 123/122.88 -- Transmit raster period in us. Will be overwritten if the PulSeq file includes a "tx_t" in the definitions.

grad_t (float): Default 1229/122.88 -- Gradient raster period in us. Will be overwritten if the PulSeq file includes a "grad_t" in the definitions.

tx_warmup (float): Default 500 -- Warmup time to turn on tx_gate before Tx events in us. Will be overwritten if the PulSeq file includes a "tx_warmup" in the definitions.

tx_zero_end (bool): Default True -- Force zero at the end of RF shapes

grad_zero_end (bool): Default True -- Force zero at the end of Gradient/Trap shapes

log_file (str): Default 'ps_interpreter' -- File (.log appended) to write run log into

log_level (int): Default 20 (INFO) -- Logger level, 0 for all, 20 to ignore debug.

# Outputs
dict: tuple of numpy.ndarray time and update arrays, with variable name keys

dict: parameter dictionary containing raster times, readout numbers, and any file-defined variables

