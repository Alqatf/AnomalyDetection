import copy
import nslkdd.data.model as model

attack_types = model.attack_types
protocol_types = model.protocol_types

def get_attack_type_by_name(name):
    return attack_types.index(name)

def get_protocol_type_by_name(name):
    return protocol_types.index(name)

def df_by_attack_and_protocol_type(df, attack_type, protocol_type) :
    assert(type(attack_type) == int)
    assert(type(protocol_type) == int)
    df = df_by_attack_type(df, attack_type) 
    df = df_by_protocol_type(df, attack_type) 
    return df

def df_by_attack_type(df, attack_type) : 
    assert(type(attack_type) == int)
    df_copy = copy.deepcopy(df)
    # only select for 1 type
    df_copy = df_copy[(df_copy["attack"] == attack_type)]
    return df_copy

def df_by_protocol_type(df, protocol_type) : 
    assert(type(protocol_type) == int)
    df_copy = copy.deepcopy(df)
    # only select for 1 type
    df_copy = df_copy[(df_copy["protocol_type"] == protocol_type)] 
    return df_copy

