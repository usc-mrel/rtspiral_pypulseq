from pypulseq.make_trapezoid import make_trapezoid

def kernel_crusher(seq, prep_param, params, system):
    thickness = params['acquisition']['slice_thickness'] * 1e-3 # [mm] -> [m]
    crusher = make_trapezoid(channel=prep_param['channel'], area=prep_param['cycles_per_thickness']/thickness, system=system)
    seq.add_block(crusher)