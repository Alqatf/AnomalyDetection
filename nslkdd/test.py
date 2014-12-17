import preprocessing as preprocessing
import data.model as model


if __name__ == '__main__':
    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()
    
    # it shows every headers
    df = df_training_20
    df_cut = df[0:1]
    for i, r in df_cut.iterrows() :
        message = ""
        for header in headers:
            print header + str(r[header])
            message = message + str(r[header]) + "/"
        print str(i) + ": " + message 

#    while True:
#        command = "1:"
#        n = raw_input(command)
#        if int(n) == 1:
#        elif int(n) == 2:
#        else :
#            pass

