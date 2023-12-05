from kernels.kernel_tag_CSPAMM import kernel_tag_CSPAMM
from kernels.kernel_rf import kernel_rf
from kernels.kernel_crusher import kernel_crusher

def kernel_handle_preparations(seq, param, system):
    # handle any preparation pulses.
    # see if param['preparations'] exists.
    if 'preparations' in param:
        for prep in param['preparations']:
            if prep['type'] == 'tagging':
                kernel_tag_CSPAMM(seq, system)
            elif prep['type'] == 'rf':
                kernel_rf(seq, prep, param, system)
            elif prep['type'] == 'crusher':
                kernel_crusher(seq, prep, system)


def kernel_handle_end_preparations(seq, param, system):
    # handle any end_preparation pulses.
    # see if param['end_preparations'] exists.
    if 'end_preparations' in param:
        for prep in param['end_preparations']:
            if prep['type'] == 'tagging':
                kernel_tag_CSPAMM(seq, system)
            elif prep['type'] == 'rf':
                kernel_rf(seq, prep, param, system)
            elif prep['type'] == 'crusher':
                kernel_crusher(seq, prep, system)