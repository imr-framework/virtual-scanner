# -*- coding: utf-8 -*-
# pulseq_assembler.py
# Written by Lincoln Craven-Brightman

import numpy as np
import logging  # For errors


class PSInterpreter:
    """
    Interpret object that can compile a PulSeq file into a FLOCRA update stream array.
    Run PSInterpreter.compile to compile a .seq file into a [updates]x[variables]

    Attributes:
        out_dict (complex): Output sequence data
        readout_number (int): Expected number of readouts
    """

    def __init__(self, rf_center=3e+6, rf_amp_max=5e+3, grad_max=1e+7,
                 gx_max=None, gy_max=None, gz_max=None,
                 clk_t=1 / 122.88, tx_t=123 / 122.88, grad_t=1229 / 122.88,
                 tx_warmup=500, tx_zero_end=True, grad_zero_end=True,
                 log_file='ps_interpreter', log_level=20):
        """
        Create PSInterpreter object for FLOCRA with system parameters.

        Args:
            rf_center (float): RF center (local oscillator frequency) in Hz.
            rf_amp_max (float): Default 5e+3 -- System RF amplitude max in Hz.
            grad_max (float): Default 1e+6 -- System gradient max in Hz/m.
            gx_max (float): Default None -- System X-gradient max in Hz/m. If None, defaults to grad_max.
            gy_max (float): Default None -- System Y-gradient max in Hz/m. If None, defaults to grad_max.
            gz_max (float): Default None -- System Z-gradient max in Hz/m. If None, defaults to grad_max.
            clk_t (float): Default 1/122.88 -- System clock period in us.
            tx_t (float): Default 123/122.88 -- Transmit raster period in us.
            grad_t (float): Default 1229/122.88 -- Gradient raster period in us.
            tx_warmup (float): Default 500 -- Warmup time to turn on tx_gate before Tx events in us.
            tx_zero_end (bool): Default True -- Force zero at the end of RF shapes
            grad_zero_end (bool): Default True -- Force zero at the end of Gradient/Trap shapes
            log_file (str): Default 'ps_interpreter' -- File (.log appended) to write run log into.
            log_level (int): Default 20 (INFO) -- Logger level, 0 for all, 20 to ignore debug.
        """
        # Logging
        self._logger = logging.getLogger()
        logging.basicConfig(filename=log_file, filemode='w', level=log_level)

        self._clk_t = clk_t  # Instruction clock period in us
        self._tx_t = tx_t  # Transmit sample period in us
        self._warning_if(int(tx_t / self._clk_t) * self._clk_t != tx_t,
                         f"tx_t ({tx_t}) isn't a multiple of clk_t ({clk_t})")
        self._grad_t = grad_t  # Gradient sample period in us
        self._warning_if(int(grad_t / self._clk_t) * self._clk_t != grad_t,
                         f"grad_t ({(grad_t)}) isn't multiple of clk_t ({clk_t})")
        self._rx_div = None
        self._rx_t = None

        self._rf_center = rf_center  # Hz
        self._rf_amp_max = rf_amp_max  # Hz

        # Gradient maxes, Hz/m
        self._grad_max = {}
        if gx_max is None:
            self._grad_max['gx'] = grad_max
        else:
            self._grad_max['gx'] = gx_max
        if gx_max is None:
            self._grad_max['gy'] = grad_max
        else:
            self._grad_max['gy'] = gy_max
        if gx_max is None:
            self._grad_max['gz'] = grad_max
        else:
            self._grad_max['gz'] = gz_max

        self._tx_warmup = tx_warmup  # us

        self._tx_zero_end = tx_zero_end
        self._grad_zero_end = grad_zero_end

        # Interpreter for section names in .seq file
        self._pulseq_keys = {
            '[VERSION]': self._read_temp,  # Unused
            '[DEFINITIONS]': self._read_defs,
            '[BLOCKS]': self._read_blocks,
            '[RF]': self._read_rf_events,
            '[GRADIENTS]': self._read_grad_events,
            '[TRAP]': self._read_trap_events,
            '[ADC]': self._read_adc_events,
            '[DELAYS]': self._read_delay_events,
            '[EXTENSIONS]': self._read_temp,  # Unused
            '[SHAPES]': self._read_shapes
        }

        # Defined variable names to output
        self._var_names = ('tx0', 'grad_vx', 'grad_vy', 'grad_vz', 'grad_vz2',
                           'rx0_en', 'tx_gate')

        # PulSeq dictionary storage
        self._blocks = {}
        self._rf_events = {}
        self._grad_events = {}
        self._adc_events = {}
        self._delay_events = {}
        self._shapes = {}
        self._definitions = {}

        # Interpolated and compiled data for output
        self._tx_durations = {}  # us
        self._tx_times = {}  # us
        self._tx_data = {}  # normalized float
        self._grad_durations = {}  # us
        self._grad_times = {}  # us
        self._grad_data = {}  # normalized float

        self.out_data = {}
        self.readout_number = 0
        self.is_assembled = False

    # Wrapper for full compilation
    def interpret(self, pulseq_file):
        """
        Interpret FLOCRA array from PulSeq .seq file

        Args:
            pulseq_file (str): PulSeq file to compile from

        Returns:
            dict: tuple of numpy.ndarray time and update arrays, with variable name keys
            dict: parameter dictionary containing raster times, readout numbers, and any file-defined variables
        """
        self._logger.info(f'Interpreting ' + str(pulseq_file))
        if self.is_assembled:
            self._logger.info('Re-initializing over old sequence...')
            self.__init__(rf_center=self._rf_center, rf_amp_max=self._rf_amp_max,
                          gx_max=self._grad_max['gx'], gy_max=self._grad_max['gy'], gz_max=self._grad_max['gz'],
                          clk_t=self._clk_t, tx_t=self._tx_t, grad_t=self._grad_t)
        self._read_pulseq(pulseq_file)
        self._compile_tx_data()
        self._compile_grad_data()
        self.out_data, self.readout_number = self._stream_all_blocks()
        self.is_assembled = True
        param_dict = {'readout_number': self.readout_number, 'tx_t': self._tx_t, 'rx_t': self._rx_t,
                      'grad_t': self._grad_t}
        for key, value in self._definitions.items():
            if key in param_dict:
                self._logger.warning(
                    f'Key conflict: overwriting key [{key}], value [{param_dict[key]}] with new value [{value}]')
            param_dict[key] = value
        return self.out_data, param_dict

    # Open file and read in all sections into class storage
    def _read_pulseq(self, pulseq_file):
        """
        Read PulSeq file into object dict memory

        Args:
            pulseq_file (str): PulSeq file to assemble from
        """
        # Open file
        with open(pulseq_file) as f:
            self._logger.info('Opening PulSeq file...')
            line = '\n'
            next_line = ''

            while True:
                if not next_line:
                    line = f.readline()
                else:
                    line = next_line
                    next_line = ''
                if line == '': break
                key = self._simplify(line)
                if key in self._pulseq_keys:
                    next_line = self._pulseq_keys[key](f)

        # Check that all ids are valid
        self._logger.info('Validating ids...')
        var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
        var_dicts = [self._delay_events, self._rf_events, self._grad_events, self._grad_events, self._grad_events,
                     self._adc_events, {}]
        for block in self._blocks.values():
            for i in range(len(var_names)):
                id_n = block[var_names[i]]
                self._error_if(id_n != 0 and id_n not in var_dicts[i], f'Invalid {var_names[i]} id: {id_n}')
        for rf in self._rf_events.values():
            self._error_if(rf['mag_id'] not in self._shapes, f'Invalid magnitude shape id: {rf["mag_id"]}')
            self._error_if(rf['phase_id'] not in self._shapes, f'Invalid phase shape id: {rf["phase_id"]}')
        for grad in self._grad_events.values():
            if len(grad) == 3:
                self._error_if(grad['shape_id'] not in self._shapes, f'Invalid grad shape id: {grad["shape_id"]}')
        self._logger.info('Valid ids')

        # Check that all delays are multiples of clk_t
        for events in [self._blocks.values(), self._rf_events.values(), self._grad_events.values(),
                       self._adc_events.values()]:
            for event in events:
                self._warning_if(int(event['delay'] / self._clk_t) * self._clk_t != event['delay'],
                                 f'Event delay {event["delay"]} is not a multiple of clk_t')
        for delay in self._delay_events.values():
            self._warning_if(int(delay / self._clk_t) * self._clk_t != delay,
                             f'Delay event {delay} is not a multiple of clk_t')

        # Check that RF/ADC (TX/RX) only have one frequency offset -- can't be set within one file.
        freq = None
        base_id = None
        base_str = None
        for rf_id, rf in self._rf_events.items():
            if freq is None:
                freq = rf['freq']
                base_id = rf_id
                base_str = 'RF'
            self._error_if(rf['freq'] != freq,
                           f"Frequency offset of RF event {rf_id} ({rf['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
        for adc_id, adc in self._adc_events.items():
            if freq is None:
                freq = adc['freq']
                base_id = adc_id
                base_str = 'ADC'
            self._error_if(adc['freq'] != freq,
                           f"Frequency offset of ADC event {adc_id} ({adc['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
        if freq is not None and freq != 0:
            self._rf_center += freq
            self._logger.info(
                f'Adding freq offset {freq} Hz. New center / linear oscillator frequency: {self._rf_center}')

        # Check that ADC has constant dwell time
        dwell = None
        for adc_id, adc in self._adc_events.items():
            if dwell is None:
                dwell = adc['dwell'] / 1000
                base_id = adc_id
            self._error_if(adc['dwell'] / 1000 != dwell,
                           f"Dwell time of ADC event {adc_id} ({adc['dwell']}) doesn't match that of ADC event {base_id} ({dwell})")
        if dwell is not None:
            self._rx_div = np.round(dwell / self._clk_t).astype(int)
            self._rx_t = self._clk_t * self._rx_div
            self._warning_if(self._rx_div * self._clk_t != dwell,
                             f'Dwell time ({dwell}) rounded to {self._rx_t}, multiple of clk_t ({self._clk_t})')

        self._logger.info('PulSeq file loaded')

    # Compilation into data formats
    # region

    # Interpolate and compile tx events
    def _compile_tx_data(self):
        """
        Compile transmit data from object dict memory into concatenated array
        """

        self._logger.info('Compiling Tx data...')

        # Process each rf event
        for tx_id, tx_event in self._rf_events.items():
            # Collect mag/phase shapes
            mag_shape = self._shapes[tx_event['mag_id']]
            phase_shape = self._shapes[tx_event['phase_id']]
            self._error_if(len(mag_shape) != len(phase_shape),
                           f'Tx envelope of RF event {tx_id} has mismatched magnitude ' \
                           'and phase length')

            # Event length and duration, create time points
            event_len = len(mag_shape)  # unitless
            event_duration = event_len * self._tx_t  # us
            self._error_if(event_len < 1, f"Zero length shape: {tx_event['mag_id']}")
            x = np.linspace(0, event_duration, num=event_len, endpoint=False)

            # Scale and convert to complex Tx envelope
            mag = mag_shape * tx_event['amp'] / self._rf_amp_max
            phase = phase_shape * 2 * np.pi
            tx_env = np.exp((phase + tx_event['phase']) * 1j) * mag

            self._error_if(np.any(np.abs(tx_env) > 1.0), f'Magnitude of RF event {tx_id} is too ' \
                                                         f'large relative to RF max {self._rf_amp_max}')

            # Optionally force zero at the end of tx event
            if self._tx_zero_end:
                x = np.append(x, event_duration)
                tx_env = np.append(tx_env, 0)

            # Save tx duration, update times, data
            self._tx_durations[tx_id] = event_duration + tx_event['delay']
            self._tx_times[tx_id] = x + tx_event['delay']
            self._tx_data[tx_id] = tx_env

        self._logger.info('Tx data compiled')

    # Interpolate and compile gradient events
    def _compile_grad_data(self):
        """
        Compile gradient events from object dict memory into array
        """
        self._logger.info('Compiling gradient data...')

        # Process each rf event
        for grad_id, grad_event in self._grad_events.items():

            # Collect shapes, create time points
            if len(grad_event) == 5:  # Trapezoid shape

                # Check for timing issues
                for time in ['rise', 'flat', 'fall']:
                    self._warning_if(grad_event[time] < self._grad_t, f'Trapezoid {grad_id} has {time} ' \
                                                                      f"time ({grad_event[time]}) less than raster time ({self._grad_t})")
                    self._warning_if(int(grad_event[time] / self._grad_t) * self._grad_t != grad_event[time],
                                     f"Trapezoid {grad_id} {time} time ({grad_event[time]}) isn't a multiple of raster time ({self._grad_t})")

                # Raster out rise and fall times, prioritize flat time and zero ending
                rise_len = int(grad_event['rise'] / self._grad_t)
                fall_len = int(grad_event['fall'] / self._grad_t)

                x_rise = np.linspace(grad_event['rise'] - rise_len * self._grad_t,
                                     grad_event['rise'],
                                     num=rise_len, endpoint=False)
                rise = np.flip(np.linspace(grad_event['amp'], 0, num=rise_len, endpoint=False))

                x_fall = np.linspace(grad_event['rise'] + grad_event['flat'],
                                     grad_event['rise'] + grad_event['flat'] + fall_len * self._grad_t,
                                     num=fall_len, endpoint=False)
                fall = np.flip(np.linspace(0, grad_event['amp'], num=fall_len, endpoint=False))

                # Concatenate times and data
                x = np.concatenate((x_rise, x_fall))
                grad = np.concatenate((rise, fall))

                event_duration = grad_event['rise'] + grad_event['flat'] + grad_event['fall']  # us
            else:
                # Event length and duration, create time points
                shape = self._shapes[grad_event['shape_id']]
                event_len = len(shape)  # unitless
                event_duration = event_len * self._grad_t  # us
                self._error_if(event_len < 1, f"Zero length shape: {grad_event['shape_id']}")
                grad = shape * grad_event['amp']
                x = np.linspace(0, event_duration, num=event_len, endpoint=False)

            # Optionally force zero at the end of gradient event
            if self._grad_zero_end:
                x = np.append(x, event_duration)
                grad = np.append(grad, 0)

            # Save grad duration, update times, data
            self._grad_durations[grad_id] = event_duration + grad_event['delay']
            self._grad_times[grad_id] = x + grad_event['delay']
            self._grad_data[grad_id] = grad

        self._logger.info('Gradient data compiled')

    # Encode all blocks
    def _stream_all_blocks(self):
        """
        Encode all blocks into sequential time updates.

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            int: number of sequence readout points
        """
        # Prep containers, zero at start
        out_data = {}
        times = {var: [np.zeros(1)] for var in self._var_names}
        updates = {var: [np.zeros(1)] for var in self._var_names}
        start = 0
        readout_total = 0

        # Encode all blocks
        for block_id in self._blocks.keys():
            var_dict, duration, readout_num = self._stream_block(block_id)

            for var in self._var_names:
                times[var].append(var_dict[var][0] + start)
                updates[var].append(var_dict[var][1])

            start += duration
            readout_total += readout_num

        # Clean up final arrays
        for var in self._var_names:
            # Make sure times are ordered, and overwrite duplicates to last inserted update
            time_sorted, unique_idx = np.unique(np.flip(np.concatenate(times[var])), return_index=True)
            update_sorted = np.flip(np.concatenate(updates[var]))[unique_idx]

            # Compressed repeated values
            update_compressed_idx = np.concatenate([[0], np.nonzero(update_sorted[1:] - update_sorted[:-1])[0] + 1])
            update_arr = update_sorted[update_compressed_idx]
            time_arr = time_sorted[update_compressed_idx]

            # Zero everything at end
            time_arr = np.concatenate((time_arr, np.zeros(1) + start))
            update_arr = np.concatenate((update_arr, np.zeros(1)))

            out_data[var] = (time_arr, update_arr)

        return (out_data, readout_total)

    # Convert individual block into PR commands (duration, gates), TX offset, and GRAD offset
    def _stream_block(self, block_id):
        """
        Encode block into sequential time updates

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            float: duration of the block
            int: readout count for the block
        """
        out_dict = {var: [] for var in self._var_names}
        readout_num = 0
        duration = 0

        block = self._blocks[block_id]
        # Preset all variables
        for var in self._var_names:
            out_dict[var] = (np.zeros(0, dtype=int),) * 2

        # Minimum duration of block
        if block['delay'] != 0:
            duration = max(duration, self._delay_events[block['delay']])

        # Tx and Tx gate updates
        tx_id = block['rf']
        if tx_id != 0:
            out_dict['tx0'] = (self._tx_times[tx_id], self._tx_data[tx_id])
            duration = max(duration, self._tx_durations[tx_id])
            tx_gate_start = self._tx_times[tx_id][0] - self._tx_warmup
            self._error_if(tx_gate_start < 0,
                           f'Tx warmup ({self._tx_warmup}) of RF event {tx_id} is longer than delay ({self._tx_times[tx_id][0]})')
            out_dict['tx_gate'] = (np.array([tx_gate_start, self._tx_durations[tx_id]]),
                                   np.array([1, 0]))

        # Gradient updates
        for grad_ch in ('gx', 'gy', 'gz'):
            grad_id = block[grad_ch]
            if grad_id != 0:
                grad_var_name = grad_ch[0] + 'rad_v' + grad_ch[
                    1]  # To get the correct varname for output g[CH] -> grad_v[CH]
                self._error_if(np.any(np.abs(self._grad_data[grad_id] / self._grad_max[grad_ch]) > 1),
                               f'Gradient event {grad_id} for {grad_ch} in block {block_id} is larger than {grad_ch} max')
                out_dict[grad_var_name] = (
                    self._grad_times[grad_id], self._grad_data[grad_id] / self._grad_max[grad_ch])
                duration = max(duration, self._grad_durations[grad_id])

        # Rx updates
        rx_id = block['adc']
        if rx_id != 0:
            rx_event = self._adc_events[rx_id]
            rx_start = rx_event['delay']
            rx_end = rx_start + rx_event['num'] * self._rx_t
            readout_num += rx_event['num']
            out_dict['rx0_en'] = (np.array([rx_start, rx_end]), np.array([1, 0]))
            duration = max(duration, rx_end)

        # Return durations for each PR and leading edge values
        return (out_dict, duration, int(readout_num))

    # endregion

    # Helper functions for reading sections
    # region

    # [BLOCKS] <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>
    def _read_blocks(self, f):
        """
        Read BLOCKS (event block) section in PulSeq file f to object dict memory.
        Event blocks are formatted like: <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
        rline = ''
        line = ''
        self._logger.info('Blocks: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 8:  # <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>
                data_line = [int(x) for x in tmp]
                self._warning_if(data_line[0] in self._blocks, f'Repeat block ID {data_line[0]}, overwriting')
                self._blocks[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}
            elif len(tmp) == 7:  # Spec allows extension ID not included, add it in as 0
                data_line = [int(x) for x in tmp]
                data_line.append(0)
                self._warning_if(data_line[0] in self._blocks, f'Repeat block ID {data_line[0]}, overwriting')
                self._blocks[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}

        if len(self._blocks) == 0: self._logger.error('Zero blocks read, nonzero blocks needed')
        assert len(self._blocks) > 0, 'Zero blocks read, nonzero blocks needed'
        self._logger.info('Blocks: Complete')

        return rline

    # [RF] <id> <amp> <mag_id> <phase_id> <delay> <freq> <phase>
    def _read_rf_events(self, f):
        """
        Read RF (RF event) section in PulSeq file f to object dict memory.
        RF events are formatted like: <id> <amp> <mag_id> <phase_id> <delay> <freq> <phase>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'mag_id', 'phase_id', 'delay', 'freq', 'phase')
        rline = ''
        line = ''
        self._logger.info('RF: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 7:  # <id> <amp> <mag id> <phase id> <delay> <freq> <phase>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5]),
                             float(tmp[6])]
                self._warning_if(data_line[0] in self._rf_events, f'Repeat RF ID {data_line[0]}, overwriting')
                self._rf_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}

        self._logger.info('RF: Complete')

        return rline

    # [GRADIENTS] <id> <amp> <shape_id> <delay>
    def _read_grad_events(self, f):
        """
        Read GRADIENTS (gradient event) section in PulSeq file f to object dict memory.
        Gradient events are formatted like: <id> <amp> <shape_id> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'shape_id', 'delay')
        rline = ''
        line = ''
        self._logger.info('Gradients: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 4:  # GRAD <id> <amp> <shape id> <delay>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3])]
                self._warning_if(data_line[0] in self._grad_events,
                                 f'Repeat gradient ID {data_line[0]} in GRADIENTS, overwriting')
                self._grad_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}
            elif len(tmp) == 3:  # GRAD <id> <amp> <shape id> NO DELAY
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2])]
                data_line.append(0)
                self._warning_if(data_line[0] in self._grad_events,
                                 f'Repeat gradient ID {data_line[0]}, in GRADIENTS, overwriting')
                self._grad_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}

        self._logger.info('Gradients: Complete')

        return rline

    # [TRAP] <id> <amp> <rise> <flat> <fall> <delay>
    def _read_trap_events(self, f):
        """
        Read TRAP (trapezoid gradient event) section in PulSeq file f to object dict memory.
        Trapezoid gradient events are formatted like: <id> <amp> <rise> <flat> <fall> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'rise', 'flat', 'fall', 'delay')
        rline = ''
        line = ''
        self._logger.info('Trapezoids: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 6:  # TRAP <id> <amp> <rise> <flat> <fall> <delay>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5])]
                self._warning_if(data_line[0] in self._grad_events,
                                 f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
                self._grad_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}
            elif len(tmp) == 5:  # TRAP <id> <amp> <rise> <flat> <fall> NO DELAY
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])]
                data_line.append(0)
                self._warning_if(data_line[0] in self._grad_events,
                                 f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
                self._grad_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}

        self._logger.info('Trapezoids: Complete')

        return rline

    # [ADC] <id> <num> <dwell> <delay> <freq> <phase>
    def _read_adc_events(self, f):
        """
        Read ADC (ADC/readout event) section in PulSeq file f to object dict memory.
        ADC events are formatted like: <id> <num> <dwell> <delay> <freq> <phase>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('num', 'dwell', 'delay', 'freq', 'phase')
        rline = ''
        line = ''
        self._logger.info('ADC: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 6:
                data_line = [int(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3]), float(tmp[4]), float(tmp[5])]
                self._adc_events[data_line[0]] = {var_names[i]: data_line[i + 1] for i in range(len(var_names))}

        self._logger.info('ADC: Complete')

        return rline

    # [DELAY] <id> <delay> -> single value output
    def _read_delay_events(self, f):
        """
        Read DELAY (delay event) section in PulSeq file f to object dict memory (stored as a single value, not a dict).
        Delay events are formatted like: <id> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Delay: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 2:
                data_line = [int(x) for x in tmp]
                self._warning_if(data_line[0] in self._delay_events, f'Repeat delay ID {data_line[0]}, overwriting')
                self._delay_events[data_line[0]] = data_line[1]  # Single value, delay

        self._logger.info('Delay: Complete')

        return rline

    # [SHAPES] list of entries, normalized between 0 and 1
    def _read_shapes(self, f):
        """
        Read SHAPES (rastered shapes) section in PulSeq file f to object dict memory.
        Shapes are formatted with two header lines, followed by lines of single data points in compressed pulseq shape format

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Shapes: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break
            if len(rline.split()) == 2 and rline.split()[0].lower() == 'shape_id':
                shape_id = int(rline.split()[1])
                n = int(self._simplify(f.readline()).split()[1])
                self._warning_if(shape_id in self._shapes, f'Repeat shape ID {shape_id}, overwriting')
                self._shapes[shape_id] = np.zeros(n)
                i = 0
                prev = -2
                x = 0
                while i < n:
                    dx = float(self._simplify(f.readline()))
                    x += dx
                    self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x},'
                                                     ' outside of [0, 1], will be capped')
                    if x > 1:
                        x = 1
                    elif x < 0:
                        x = 0
                    self._shapes[shape_id][i] = x
                    if dx == prev:
                        r = int(self._simplify(f.readline()))
                        for _ in range(0, r):
                            i += 1
                            x += dx
                            self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x},'
                                                             ' outside of [0, 1], will be capped')
                            if x > 1:
                                x = 1
                            elif x < 0:
                                x = 0
                            self._shapes[shape_id][i] = x
                    i += 1
                    prev = dx

        self._logger.info('Shapes: Complete')

        return rline

    # [DEFINITIONS] <varname> <value>
    def _read_defs(self, f):
        """
        Read through DEFINITIONS section in PulSeq file f.

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Definitions: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 2:
                varname, value = rline.split()
                try:
                    value = float(value)
                except:
                    pass

                # Automatic raster time reading
                if varname == 'tx_t':
                    self._tx_t = value
                    self._logger.info(f'Overwriting tx_t to {value} from Definitions')
                elif varname == 'grad_t':
                    self._grad_t = value
                    self._logger.info(f'Overwriting grad_t to {value} from Definitions')
                elif varname == 'tx_warmup':
                    self._tx_warmup = value
                    self._logger.info(f'Overwriting tx_warmup to {value} from Definitions')
                else:
                    self._definitions[varname] = value

                self._logger.debug(f'Read in {varname}')

        self._logger.info('Definitions: Complete')

        return rline

    # Unused headers
    def _read_temp(self, f):
        """
        Read through any unused section in PulSeq file f.

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('(Unused): Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break
            self._logger.debug('Unused line')

        self._logger.info('(Unused): Complete')

        return rline

    # Simplify lines read from pulseq -- remove comments, trailing \n, trailing whitespace, commas
    def _simplify(self, line):
        """
        Simplify raw line to space-separated values

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Simplified string
        """

        # Find and remove comments, comma
        comment_index = line.find('#')
        if comment_index >= 0:
            line = line[:comment_index]

        return line.rstrip('\n').strip().replace(',', '')

    # endregion

    # Error and warnings
    # region
    # For crashing and logging errors (may change behavior)
    def _error_if(self, err_condition, message):
        """
        Throw an error (currently using assert) and log if error condition is met

        Args:
            err_condition (bool): Condition on which to throw error
            message (str): Message to accompany error in log.
        """
        if err_condition: self._logger.error(message)
        assert not err_condition, (message)

    # For warnings without crashing
    def _warning_if(self, warn_condition, message):
        """
        Print warning and log if error condition is met

        Args:
            warn_condition (bool): Condition on which to warn
            message (str): Message to accompany warning in log.
        """
        if warn_condition: self._logger.warning(message)
    # endregion


# Sample usage
if __name__ == '__main__':
    ps = PSInterpreter(grad_t=1)
    inp_file = '../mgh-flocra/test_sequences/tabletop_radial_v2_2d_pulseq.seq'
    out_data, params = ps.interpret(inp_file)

    import matplotlib.pyplot as plt

    names = [' tx', ' gx', ' gy', ' gz', 'adc']
    data = [out_data['tx0'], out_data['grad_vx'], out_data['grad_vy'], out_data['grad_vz'], out_data['tx_gate']]

    for i in range(5):
        print(f'{names[i]} minimum entry difference magnitude: {np.min(np.abs(data[i][1][1:] - data[i][1][:-1]))}')
        print(f'{names[i]} entries below 1e-6 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-6)}')
        print(f'{names[i]} entries below 1e-5 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-5)}')
        print(f'{names[i]} entries below 1e-4 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-4)}')
        print(f'{names[i]} entries below 1e-3 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-3)}')

    print("Completed successfully")
