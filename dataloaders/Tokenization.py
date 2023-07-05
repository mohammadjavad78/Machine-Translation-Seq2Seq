def Tokenization(line):
    Tokens=['<sos>']
    line=line.lower()
    unused=['!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','\t','\n']
    for i in unused:
        line=line.replace(i,'')
    for i in line.split(' '):
        Tokens.append(i)
    Tokens.append('<eos>')
    return Tokens



# if(__name__=="__main__"):
#     print(Tokenization("Hello hOw !are you?"))

# import chakin
# chakin.search(lang='English')

# chakin.download(number=12,save_dir='./')