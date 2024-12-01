def select_network(info_net):
    name = info_net['name']
    params = info_net['params']

    if name.upper() == 'vpn'.upper():
        from net.pansharping.VPN.model_arch import VIRAttResUNetSR as Net
        
    network = Net(**params)
    return network
