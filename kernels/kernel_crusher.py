from pypulseq.make_trapezoid import make_trapezoid

def kernel_crusher(seq, prep_param, system):
    crusher = make_trapezoid(channel=prep_param['channel'], area=prep_param['cycles_per_cm']*100, system=system)
    seq.add_block(crusher)