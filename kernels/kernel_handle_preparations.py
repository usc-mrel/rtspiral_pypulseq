from kernels.kernel_tag_CSPAMM import kernel_tag_CSPAMM
from kernels.kernel_rf import kernel_rf
from kernels.kernel_crusher import kernel_crusher
from kernels.kernel_tag_radial_AM import kernel_tag_radial_AM
from kernels.kernel_trigger import kernel_trigger


def prep_func(prep_table_str, seq, param, system):
    output_string = ''
    if prep_table_str in param:
        for prep in param[prep_table_str]:
            if prep['enabled'] == True:
                output_string += prep['type'] + '_'
                if prep['type'] == 'trigger':
                    kernel_trigger(seq, prep, param, system)
                if prep['type'] == 'tagging':
                    if prep['tag_type'] == 'grid':
                        kernel_tag_CSPAMM(seq, prep, param, system)
                    if prep['tag_type'] == 'radial':
                        kernel_tag_radial_AM(seq, prep, param, system)
                elif prep['type'] == 'rf':
                    kernel_rf(seq, prep, param, system)
                elif prep['type'] == 'crusher':
                    kernel_crusher(seq, prep, param, system) 
        # remove last underscore
        output_string = output_string[:-1]
        return output_string
    else:
        return output_string

def kernel_handle_preparations(seq, param, system):
    return prep_func('preparations', seq, param, system)
  

def kernel_handle_end_preparations(seq, param, system):
    return prep_func('end_preparations', seq, param, system)
  