import copy
import nslkdd.data.model as model

attack_types = model.attack_types
protocol_types = model.protocol_types

def df_by_attack_and_protocol_type(df, attack_type, protocol_type) : 
    assert(type(attack_type) == int)
    assert(type(protocol_type) == int)
    df_copy = copy.deepcopy(df)
    df_copy = df_copy[(df_copy["attack"] == attack_type)] # only select for 1 class 
    df_copy = df_copy[(df_copy["protocol_type"] == protocol_type)]
    return df_copy

