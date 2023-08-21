from pypulseq.Sequence.sequence import Sequence
import matplotlib.pyplot as plt
import h5py

# Load both files


q1 = 3
q2 = 6

seq_orig = Sequence()
seq_orig.read('gre_jemris.seq')
print('Seq original')
print(seq_orig.get_block(q1).gx)



seq_proc = Sequence()
seq_proc.read('gre_jemris_seq2xml_jemris.seq')

#print("Seq processed:")
#print(seq_proc.get_block(q2).gx)

#seq_orig.plot(time_range=[0,10])

#seq_proc.plot(time_range=[0,10])


sd = h5py.File('gre_jemris_seq2xml_jemris.h5','r')
sd = sd['seqdiag']
plt.figure(1)
plt.subplot(411)
plt.title("Twice converted JEMRIS sequence diagram")

plt.plot(sd['T'][()],sd['TXM'][()])
plt.subplot(412)
plt.plot(sd['T'][()],sd['GX'][()])
plt.subplot(413)
plt.plot(sd['T'][()],sd['GY'][()])
plt.subplot(414)
plt.plot(sd['T'][()],sd['GZ'][()])




sd = h5py.File('gre.h5','r')
sd = sd['seqdiag']
plt.figure(2)

plt.subplot(411)
plt.title("Original JEMRIS sequence diagram")

plt.plot(sd['T'][()],sd['TXM'][()])
plt.subplot(412)
plt.plot(sd['T'][()],sd['GX'][()])
plt.subplot(413)
plt.plot(sd['T'][()],sd['GY'][()])
plt.subplot(414)
plt.plot(sd['T'][()],sd['GZ'][()])




plt.show()