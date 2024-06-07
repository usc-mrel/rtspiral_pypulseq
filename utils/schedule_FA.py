import numpy as np
from utils.calculate_ramp_ibrahim import calculate_ramp_ibrahim

def schedule_FA(params, n_TRs):
    if 'fa_schedule' in params['acquisition']:
        if params['acquisition']['fa_schedule']['type'] == "ramp_ibrahim":
            if params['acquisition']['fa_schedule']['enabled'] == True:
                T1 = params['acquisition']['fa_schedule']['T1'] * 1e-3
                T2 = params['acquisition']['fa_schedule']['T2'] * 1e-3
                TR = params['acquisition']['TR']
                rf_amplitudes = calculate_ramp_ibrahim(n_TRs, T1, T2, TR, np.deg2rad(params['acquisition']['flip_angle']), max_alpha=np.deg2rad(180), truncate=False)

                # pre-pend the rf_amplitudes with params['acquisition']['flip_angle']
                rf_amplitudes = np.concatenate(([np.deg2rad(params['acquisition']['flip_angle'])], rf_amplitudes))
                params['flip_angle_last'] = np.rad2deg(rf_amplitudes[-1])
                FA_schedule_str = "_ramp_ibrahim_"
                return rf_amplitudes, FA_schedule_str 
    return '', '_' 
