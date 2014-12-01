import os
from datetime import date

def delete_file(path=None):
    if path != None:
        os.remove(path)

def make_today_folder(path="./"):
    path = path+'/'+today_to_string()
    try :
        os.mkdir(path)
    except OSError :
        pass
    return path

def today_to_string():
    d = date.today()
    return d.strftime("%Y-%m-%d")
    
if __name__ == '__main__':
    make_today_folder('./results')
