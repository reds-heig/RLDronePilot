
def init_neptune(neptune_params):
    import neptune
    import yaml
    
    # load neptune api key
    with open('.neptune_config.yaml', 'r') as config_file:
        neptune_config = yaml.safe_load(config_file)
    api_token = neptune_config.get('api_token')

    # init neptune
    run = neptune.init_run(
        project='chacungu/Drone-Controller-with-Reinforcement-Learning', 
        api_token=api_token, 
        tags=['reds'],
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False,
        capture_hardware_metrics=False,
    )
    run['parameters'] = neptune_params
    return run