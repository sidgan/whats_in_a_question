#!/usr/bin/env python

"""generate data files"""

###########
# imports #
###########
import json
from collections import defaultdict
import string
from collections import OrderedDict

####################
# define variables #
####################
image_id = []
question = defaultdict(str)
multiple_choices = defaultdict(str)
question_id = defaultdict(str)
question_type = defaultdict(str)
multiple_choice_answer = defaultdict(str)
answers = defaultdict(str)
answer_type = defaultdict(str)

#####################
# String formatting #
#####################
def fix_string(s):
     return s.encode('ascii', 'ignore').translate(None, string.punctuation).lower()

###############
# Format list #
###############
def fix_multiple_choice(l):
     choices = ''
     for choice in l:
          choices += fix_string(choice) + ","
     return choices

#########################
# Process question file #
#########################
with open('MultipleChoice_mscoco_train2014_questions.json') as data_file:    
    data = json.load(data_file)

for each in data['questions']:
	image_id.append(int(each[u'image_id']))
	question[each[u'image_id']] +=  fix_string(each[u'question']) + " "
	multiple_choices[each[u'image_id']] +=  fix_multiple_choice(each[u'multiple_choices'])
	question_id[each[u'image_id']] +=  str(each[u'question_id']) + " "

#####################################
# maintain order and perform checks #
#####################################
image_id = sorted(list(set(image_id)))

assert len(image_id) == len(question_id) == len(question) == len(multiple_choices)

question =  OrderedDict(sorted(question.items(), key=lambda t: t[0]))
multiple_choices =  OrderedDict(sorted(multiple_choices.items(), key=lambda t: t[0]))
question_id =  OrderedDict(sorted(question_id.items(), key=lambda t: t[0]))

########################
# Write question files #
########################
with open('other_coco_trainval2014_train_imglist.txt','wb') as f:
          f.write('\n'.join(map(lambda x: str(x), image_id)))

with open('other_coco_trainval2014_train_question.txt','wb') as f:
     for key, value in question.items():
     	f.write(''.join(value))
     	f.write('\n')

with open('other_coco_trainval2014_train_choice.txt','wb') as f:
     for key, value in multiple_choices.items():
     	f.write(''.join(value))
     	f.write('\n')

with open('other_coco_trainval2014_train_question_id.txt','wb') as f:
     for key, value in question_id.items():
     	f.write(''.join(value))
     	f.write('\n')  

#####################
# Delete dictionary #
#### Clear memory ###
#####################
question.clear()
multiple_choices.clear()
question_id.clear()

###########################
# Process annotation file #
###########################
with open('mscoco_train2014_annotations.json') as data_file:    
    data = json.load(data_file)

for each in data['annotations']:
	imgid = int(each[u'image_id'])
	assert imgid in image_id, imgid
	question_type[imgid] += fix_string(each[u'question_type'])  + " "
	multiple_choice_answer[imgid] += fix_string(each[u'multiple_choice_answer'])  + " "
	answers[imgid] += fix_multiple_choice(each[u'answers'][i][u'answer'] for i in xrange(len(each[u'answers'])))
	answer_type[imgid] += each[u'answer_type']  + " "

#####################################
# maintain order and perform checks #
#####################################
assert len(image_id) == len(question_type) == len(multiple_choice_answer) == len(answers) == len(answer_type)

question_type = OrderedDict(sorted(question_type.items(), key=lambda t: t[0]))
multiple_choice_answer = OrderedDict(sorted(multiple_choice_answer.items(), key=lambda t: t[0]))
answers = OrderedDict(sorted(answers.items(), key=lambda t: t[0]))
answer_type =  OrderedDict(sorted(answer_type.items(), key=lambda t: t[0]))

##########################
# Write annotation files #
##########################
with open('other_coco_trainval2014_train_question_type.txt','wb') as f:
     for key, value in question_type.items():
     	f.write(''.join(value))
     	f.write('\n')

with open('other_coco_trainval2014_train_multiple_choice_answer.txt','wb') as f:
     for key, value in multiple_choice_answer.items():
     	f.write(''.join(value))
     	f.write('\n')

with open('other_coco_trainval2014_train_answers.txt','wb') as f:
     for key, value in answers.items():
     	f.write(''.join(value))
     	f.write('\n')

with open('other_coco_trainval2014_train_question_id.txt','wb') as f:
     for key, value in answer_type.items():
     	f.write(''.join(value))
     	f.write('\n')
