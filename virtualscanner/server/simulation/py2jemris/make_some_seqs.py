from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
import numpy as np

seq = Sequence(system=Opts())

gx = make_arbitrary_grad(channel='x',waveform=np.array([1,2,3,4,5,4,3,2,1]))
seq.add_block(gx)

seq.write("hiseq.seq")

seq2 = Sequence()
seq2.read('hiseq.seq')
print(seq2.get_block(1).gx)