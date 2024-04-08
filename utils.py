
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

def render_func(plot_queue):
    while True:
        env = plot_queue.get(block=True)
        env.render(out)
        
def update_pbar_func(pbar_queue, pbar):
    while True:
        pbar_queue.get() # wait for progress update
        #pbar.update(1) # TODO TMP