from kernels.kernel_tag_SPAMM_REALTAG import kernel_tag_SPAMM_REALTAG
from kernels.kernel_tag_SPAMM import kernel_tag_SPAMM
from kernels.kernel_rf import kernel_rf
from kernels.kernel_crusher import kernel_crusher
from kernels.kernel_tag_radial_AM import kernel_tag_radial_AM
from kernels.kernel_trigger import kernel_trigger


def prep_func(prep_table_str, seq, param, system, rf=None, gz=None, gzr=None):
    output_string = ''
    if prep_table_str in param:
        for prep in param[prep_table_str]:
            if prep['enabled'] == True:
                output_string += prep['type'] + '_'
                if prep['type'] == 'trigger':
                    kernel_trigger(seq, prep, param, system)
                if prep['type'] == 'tagging':
                    output_string += prep['tag_type'] + '_'
                    if prep['tag_type'] == 'grid':
                        kernel_tag_SPAMM_REALTAG(seq, prep, param, system)
                    if prep['tag_type'] == 'radial':
                        kernel_tag_radial_AM(seq, prep, param, system)
                    if prep['tag_type'] == 'SPAMM_grid':
                        kernel_tag_SPAMM(seq, prep, param, system)
                elif prep['type'] == 'rf':
                    kernel_rf(seq, prep, param, system, rf=rf, gz=gz, gzr=gzr)
                elif prep['type'] == 'crusher':
                    kernel_crusher(seq, prep, param, system) 
        # remove last underscore
        output_string = output_string[:-1]
        return output_string
    else:
        return output_string

def kernel_handle_preparations(seq, param, system, rf=None, gz=None, gzr=None):
    return prep_func('preparations', seq, param, system, rf=rf, gz=gz, gzr=gzr)
  

def kernel_handle_end_preparations(seq, param, system, rf=None, gz=None, gzr=None):
    return prep_func('end_preparations', seq, param, system, rf=rf, gz=gz, gzr=gzr)
  