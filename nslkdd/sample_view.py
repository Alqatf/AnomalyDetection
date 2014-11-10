if __name__ == '__main__':
    from data import model
    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

    headers, attacks = model.load_headers(headerfile)
    df = model.load_dataframe(datafile,headers,datasize=100)

