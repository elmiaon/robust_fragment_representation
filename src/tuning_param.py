
    
    if 'n_min_distance' in keys:
        start, stop, step = setting['n_min_distance']
    else:
        start, stop, step = 5, 51, 5
        # logger.warn(f"using default n_min_distance as {start}, {stop}, {step}")
    N_MIN_DISTANCE = np.arange(start, stop, step)
    
    # tuning parameters
    if 'beta' in keys:
        start, stop, step = setting['beta']
    else:
        start, stop, step = 50, 101, 5
        # logger.warn(f"using default beta as {start}, {stop}, {step}")
    BETA = np.arange(start, stop, step)   
    
    if 'p_min_entropy' in keys:
        start, stop, step = setting['p_min_entropy']
    else:
        start, stop, step = 0.2, 1, 9
        # logger.warn(f"using default p_min_entropy as {start}, {stop}, {step}")
    P_MIN_ENTROPY = np.linspace(start, stop, step)

    if 'p_top' in keys:
        start, stop, step = setting['p_top']
    else:
        start, stop, step = 0.3, 1, 8
        # logger.warn(f"using default p_top as {start}, {stop}, {step}")
    P_TOP = np.linspace(start, stop, step)

    if 'n_retrieve' in keys:
        start, stop, step = setting['n_retrieve']
    else:
        start, stop, step = 1, 10, 2
        # logger.warn(f"using default n_retrieve as {start}, {stop}, {step}")
    N_RETRIEVE = np.arange(start, stop, step)