
from pypulseq.make_trigger import make_trigger

def kernel_trigger(seq, prep_param, params=None, system=None):
    # first time: try trigger
    trigger = make_trigger(prep_param['trigger_type'], duration=2000e-6, system=system)
    seq.add_block(trigger)
