
def generate_model(gen_conf, train_conf):
    approach = train_conf['approach']

    model = None
    if approach == 'fc_densenet_ms':
        from .fc_densenet_ms import generate_fc_densenet_ms
        model = generate_fc_densenet_ms(gen_conf, train_conf)
    if approach == 'fc_densenet_dilated':
        from .fc_densenet_dilated import generate_fc_densenet_dilated
        model = generate_fc_densenet_dilated(gen_conf, train_conf)

    print(gen_conf['args'])
    if train_conf['num_retrain'] > 0:
        print('retraining...#' + str(train_conf['num_retrain']))

    return model


