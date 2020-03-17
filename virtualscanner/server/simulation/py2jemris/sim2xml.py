import xml.etree.ElementTree as ET
from io import BytesIO

def sim2xml(sim_name="mysimu2", seq="example.xml", phantom="sample.h5", Tx="uniform.xml", Rx="uniform.xml",
            seq_name="Sequence", sample_name="Sample", out_folder_name = None):

    root = ET.Element("simulate")
    root.set("name", "JEMRIS")

    sample = ET.SubElement(root, "sample")
    sample.set("name", sample_name)
    sample.set("uri", phantom)

    TXcoilarray = ET.SubElement(root, "TXcoilarray")
    TXcoilarray.set("uri", Tx)

    RXcoilarray = ET.SubElement(root, "RXcoilarray")
    RXcoilarray.set("uri", Rx)

    parameter = ET.SubElement(root, "parameter")
    parameter.set("RandomNoise", "0")
    parameter.set("EvolutionSteps", "0")
    parameter.set("EvolutionPrefix", "evol")
    parameter.set("ConcomitantFields", "0")

    sequence = ET.SubElement(root, "sequence")
    sequence.set("name", seq_name)
    sequence.set("uri", seq)

    model = ET.SubElement(root, "model")
    model.set("name", "Bloch")
    model.set("type", "CVODE")

    sim_tree = ET.ElementTree(root)
    sim_tree.write(out_folder_name + '/' + sim_name + '.xml')

    return sim_tree



# Fig 1. Draw diagram of what py2jemris consists of
# Fig 2. SDC Debug progress

if __name__ == '__main__':
    sim2xml(seq="gre_jemris_seq2xml.xml", phantom="sample.h5", Tx="uniform.xml", Rx="uniform.xml",
            seq_name="Sequence", sample_name="Sample", out_folder_name="try_seq2xml")