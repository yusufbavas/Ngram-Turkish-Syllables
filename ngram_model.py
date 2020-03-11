import re
from nltk import ngrams
from collections import Counter
import html2text
import random
import pickle

class ngram:
    def __init__(self, filename,n): 
        self.__data = []
        self.__n = n
        self.model = {}
        self.__string_data = self.__readfile(filename)
        print("Syllable in progress...")
        self.__data = self.syllables(self.__string_data)
        self.__create_ngram_probability()
        self.__gt = 0

    def __readfile(self,filename):
        
        print("Reading {} file...".format(filename))
        
        f = open(filename,"r",encoding="utf8",errors='ignore')
        #f = open(filename,"r")
        data = f.read()
        data = html2text.html2text(data)
        data = data.lower()
        data = re.sub(r'[^a-zçğıöşü ]', '', data)
        
        f.close()
        return data

    def __create_ngram_probability(self):

        print("n-gram model creating...")

        self.__ngram = ngrams(self.__data, self.__n)

        self.__counter = Counter(self.__ngram)
        self.GT_smoothing()
        
        t_gram = Counter(ngrams(self.__data, self.__n-1))

        if self.__n == 1:
            for key in self.__counter.keys():
                self.model[key] = self.__counter.get(key) / len(self.__counter.keys())
        else:   
            for key in self.__counter.keys():
                self.model[key] = self.__counter.get(key) / t_gram.get(key[0:-1])

        self.model['UNK'] = 0
    
    def GT_smoothing(self):

        count = 0
        gt_dict = {}
        for key,value in self.__counter.items():
            if value == 1:
                count +=1
            if gt_dict.get(value) == None:
                gt_dict[value] = 1
            else:
                gt_dict[value] = gt_dict.get(value) +1
        
        for key,value in self.__counter.items():
            if gt_dict.get(value + 1) != None : 
                self.__counter[key] = (value +1) * (gt_dict.get(value +1) / gt_dict.get(value) )
            
        self.__gt = count / len(self.__counter.keys())

    def test_perplexity(self,filename):

        test_data = self.__readfile(filename)       
        res_dict = {}
        test_data = test_data.split(" ")
        
        print("Perpleixty calculating...")

        for test in test_data:
            result = 1.0
            sly = [" "]
            sly.extend(self.syllables(test))
            sly.extend([" "])

            for i in range(len(sly) - self.__n +1):
                temp = sly[i:self.__n+i]
                if(self.model.get(tuple(temp)) == None):
                    flag = False
                    for t in temp:
                        if t in self.__data:
                            flag = True
                            break
                    # if we see the same syllabus before it means that this
                    # combination doesnt happened before so we use GT smoothing variable
                    if flag:
                        result *= self.__gt / len(self.model)
                    else:
                        self.model['UNK'] = self.model.get('UNK') +1
                        result *= (self.model.get('UNK')) / len(self.model)
                else:
                    result *= self.model.get(tuple(temp))
            if self.__n > len(sly):
                res = 0
                t = 0
                for key in self.__counter.keys():
                    if str(tuple(sly))[1:-1] in str(key) :
                        res += self.__counter.get(key)
                    if str(tuple(sly[:-1]))[1:-1] in str(key):
                        t += self.__counter.get(key)
                if (res == 0) or (t == 0):
                    flag = False
                    for t in temp:
                        if t in self.__data:
                            flag = True
                            break
                    if flag:
                        result *= self.__gt / len(self.model)
                    else:
                        self.model['UNK'] = self.model.get('UNK') +1
                        result = (self.model.get('UNK')) / len(self.model) 
                else:
                    result = res/t
            res_dict[''.join(sly[1:-1])] = (1 / result)**(1/ len(sly))

        return res_dict 

    # choose one of the 5 syllabus and produce a sentence
    def generate_sentence(self,res = (' ',)):

        print("Generate sentence:")
        result = ""
        res = self.__generate_sentence(res)
        for t in res[1:]:
            result += t
        while(True):
            res = self.__generate_sentence(res[1:])
            for t in res[self.__n-1:]:
                 result += t 
            if ' ' in result:
                break
        print(result.split(" ")[0])

    def __generate_sentence(self,syllable):

        keys = self.__counter.keys()
        count = [0]*5
        res = [(" "),(" "),(" "),(" "),(" ")]
        
        for key in keys:
            if (str(key).startswith(str(syllable)[:-2])):

                for i in range(0,5):
                    if self.__counter.get(key) > count[i]:
                        res[i:5-1] = res[i+1:5]
                        count[i:5-1] = count[i+1:5]
                        count[i] = self.__counter.get(key)
                        res[i] = key
                        break
        return random.choice(res)

    def syllables(self,word):
        
        syllables = []
        #print("Syllable in progress...")
        
        bits = ''.join(['1' if l in 'aeıioöuü' else '0' for l in word])

        seperators = (
            ('101', 1),
            ('1001', 2),
            ('10001', 3)
        )

        index, cut_start_pos = 0, 0

        while index < len(bits):

            for seperator_pattern, seperator_cut_pos in seperators:
                if bits[index:].startswith(seperator_pattern):

                    if (' ' in word[cut_start_pos:index + seperator_cut_pos]):

                        if word[cut_start_pos:index + seperator_cut_pos][0] == ' ':
                            syllables.append(' ')
                            syllables.append(word[cut_start_pos +1 :index + seperator_cut_pos])
                        else:
                            syllables.append(word[cut_start_pos:index + seperator_cut_pos - 1])
                            syllables.append(' ')
                    else:
                        syllables.append(word[cut_start_pos:index + seperator_cut_pos])

                    index += seperator_cut_pos
                    cut_start_pos = index
                    break

            index += 1

        syllables.append(word[cut_start_pos:])
        #print("syllables is completed")
        return syllables

    def save_model(self,name):
        print("n-gram model saving...")
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(self.model, handle)

    def load_model(self,name):
        with open(name + '.pickle', 'rb') as handle:
            return pickle.load(handle)

res = ngram("text2",2) #.test_perplexity("test_data")

with open("1gram.pickle","wb") as handle:
    pickle.dumps(res,a)


#res.generate_sentence()



"""res.generate_sentence()
res.generate_sentence()
res.generate_sentence()
res.generate_sentence()
res.generate_sentence()
res.generate_sentence()
res.generate_sentence()
"""
#for key in res.keys():
#    print(key)
#    print(res.get(key))